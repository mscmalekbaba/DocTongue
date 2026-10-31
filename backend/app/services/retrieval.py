"""Retrieval and grounded citation orchestration.

This module owns the runtime flow for answering a user question:

1. Retrieve top vector hits for the question.
2. Filter weak hits with lightweight grounding checks.
3. Ask the answer generator to answer from filtered excerpts.
4. Bind answer markers like [1], [2] back to retrieved hits.
5. Normalize markers at the document level so multiple chunk markers
    from the same document collapse to one citation number.

The returned ChatResponse includes the final answer text and the
document excerpts used as citations.
"""

import logging
import re
from datetime import datetime, timezone
from threading import Lock

from app.core.config import Settings
from app.models.schemas import ChatResponse, Citation
from app.services.chat_memory import ChatMemoryStore
from app.services.llm import AnswerGenerator, lexical_overlap_score
from app.services.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)
CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
ANSWER_LINE_RE = re.compile(r".*?(?:\n|$)", re.DOTALL)
AUDIT_LOG_LOCK = Lock()


def answer_question(
    question: str,
    session_id: str,
    settings: Settings,
    vector_store: VectorStore,
    answer_generator: AnswerGenerator,
    chat_memory_store: ChatMemoryStore,
    *,
    audit: bool = True,
) -> ChatResponse:
    """Answer a question using retrieval + grounded citation binding.

    Set *audit* to ``False`` to skip the plain-text audit log, which the
    offline evaluation runner does so that a test-set run does not append to
    the tracked chat log.
    """
    recent_turns = chat_memory_store.get_recent_turns(
        session_id=session_id,
        limit=settings.chat_memory_window,
    )
    retrieval_query = answer_generator.reformulate_query(question, recent_turns)
    hits = vector_store.search(retrieval_query, settings.retrieval_k)
    if not hits:
        response = ChatResponse(
            answer="I could not find an answer in the indexed documents.",
            citations=[],
            grounded=False,
        )
    else:
        top_hits = _filter_grounded_hits(retrieval_query, hits)
        if not top_hits:
            response = ChatResponse(
                answer="I could not find enough supporting evidence in the indexed documents to answer that.",
                citations=[],
                grounded=False,
            )
        else:
            grounded = True
            try:
                answer = answer_generator.generate_answer(question, [hit.content for hit in top_hits])
            except Exception as exc:
                logger.exception("LLM answer generation failed, returning evidence-only fallback.")
                answer = (
                    "I found relevant excerpts in the indexed documents, but I could not generate "
                    "a final response from the configured LLM provider."
                )
                grounded = False

            answer = _reconcile_markers_with_local_evidence(answer=answer, top_hits=top_hits)
            answer, cited_hits = _bind_answer_to_citations(
                answer=answer,
                top_hits=top_hits,
                max_citations=settings.max_answer_citations,
            )

            citations = [
                Citation(
                    document_id=hit.document_id,
                    filename=hit.filename,
                    excerpt=hit.content,
                    page_number=hit.page_number,
                    score=round(hit.score, 3),
                )
                for hit in cited_hits
            ]

            response = ChatResponse(
                answer=answer,
                citations=citations,
                grounded=grounded,
            )

    try:
        chat_memory_store.append_turn(session_id, question, response.answer)
    except Exception:
        logger.exception("Failed to append chat turn to memory for session %s", session_id)
    if audit:
        try:
            _append_chat_audit_log(
                question=question,
                answer=response.answer,
                settings=settings,
            )
        except Exception:
            logger.exception("Failed to append chat audit log for session %s", session_id)
    return response


def _append_chat_audit_log(
    question: str,
    answer: str,
    settings: Settings,
) -> None:
    """Append a timestamped Q&A entry to the plain-text audit log.

    Writes are serialised via :data:`AUDIT_LOG_LOCK` to be safe under
    concurrent requests.  The parent directory is created if absent.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = (
        f"Timestamp: {timestamp}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "---\n"
    )
    with AUDIT_LOG_LOCK:
        settings.chat_audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with settings.chat_audit_log_path.open("a", encoding="utf-8") as audit_log:
            audit_log.write(entry)


def _filter_grounded_hits(question: str, hits: list[SearchResult]) -> list[SearchResult]:
    """Keep hits that pass lexical-overlap or vector-score grounding thresholds."""
    grounded_hits: list[SearchResult] = []
    for hit in hits:
        if lexical_overlap_score(question, hit.content) >= 0.1 or hit.score >= 0.55:
            grounded_hits.append(hit)
    return grounded_hits


def _bind_answer_to_citations(
    answer: str,
    top_hits: list[SearchResult],
    max_citations: int,
) -> tuple[str, list[SearchResult]]:
    """Map answer markers to citations and normalize markers to document-level ids."""
    selected_hits: list[SearchResult] = []
    document_to_citation: dict[str, int] = {}

    # Convert model chunk markers like [4] into stable, document-level markers.
    # If [1] and [4] point to chunks from the same document, both become [1].
    def replace_marker(match: re.Match[str]) -> str:
        source_number = int(match.group(1))
        zero_based = source_number - 1
        if zero_based < 0 or zero_based >= len(top_hits):
            return match.group(0)

        hit = top_hits[zero_based]
        citation_number = document_to_citation.get(hit.document_id)
        if citation_number is None:
            if len(selected_hits) >= max_citations:
                return match.group(0)
            selected_hits.append(hit)
            citation_number = len(selected_hits)
            document_to_citation[hit.document_id] = citation_number

        return f"[{citation_number}]"

    normalized_answer = CITATION_MARKER_RE.sub(replace_marker, answer)
    if not selected_hits:
        return answer, _first_unique_documents(top_hits, max_citations)

    # Collapse adjacent duplicates such as "[1][1]" after normalization.
    normalized_answer = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", normalized_answer)

    return normalized_answer, selected_hits


def _reconcile_markers_with_local_evidence(
    answer: str,
    top_hits: list[SearchResult],
) -> str:
    """Correct single-marker lines when another retrieved hit is clearly better support.

    This helps with cases where the model chooses the wrong source number in a
    bullet line like ``- ... [1]`` despite stronger support from another retrieved
    source. Multi-marker lines are left unchanged.
    """
    if not answer or not top_hits:
        return answer

    def score_support(claim_text: str, hit: SearchResult) -> float:
        # Prioritize lexical evidence in the claim, with retrieval score as a tiebreaker.
        return (0.85 * lexical_overlap_score(claim_text, hit.content)) + (0.15 * hit.score)

    output: list[str] = []
    for line_match in ANSWER_LINE_RE.finditer(answer):
        line = line_match.group(0)
        if not line:
            continue

        markers = [int(match.group(1)) for match in CITATION_MARKER_RE.finditer(line)]
        unique_markers = {marker for marker in markers}
        if len(unique_markers) != 1:
            output.append(line)
            continue

        current_marker = next(iter(unique_markers))
        current_index = current_marker - 1
        if current_index < 0 or current_index >= len(top_hits):
            output.append(line)
            continue

        claim_text = CITATION_MARKER_RE.sub("", line).strip(" \t-•:\n")
        if len(claim_text) < 20:
            output.append(line)
            continue

        best_index = current_index
        current_score = score_support(claim_text, top_hits[current_index])
        best_score = current_score
        for index, hit in enumerate(top_hits):
            candidate_score = score_support(claim_text, hit)
            if candidate_score > best_score:
                best_score = candidate_score
                best_index = index

        # Require a margin to avoid noisy remaps.
        if best_index == current_index or (best_score - current_score) < 0.08:
            output.append(line)
            continue

        remapped_line = CITATION_MARKER_RE.sub(f"[{best_index + 1}]", line)
        output.append(remapped_line)

    return "".join(output)


def _first_unique_documents(hits: list[SearchResult], max_citations: int) -> list[SearchResult]:
    """Fallback selection: first N hits with unique document ids."""
    selected: list[SearchResult] = []
    seen_document_ids: set[str] = set()
    for hit in hits:
        if hit.document_id in seen_document_ids:
            continue
        selected.append(hit)
        seen_document_ids.add(hit.document_id)
        if len(selected) >= max_citations:
            break
    return selected