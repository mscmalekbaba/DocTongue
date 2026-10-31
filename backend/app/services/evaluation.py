"""Offline evaluation runner over a fixed, ground-truthed test set.

Replaces per-answer quality scoring, which had no ground truth and so could not
see the failure that matters most in document QA: retrieving the *wrong*
document and answering confidently from it.  An answer quoting the Premium
deductible in response to a question about the Standard plan scores highly on
answer-relevancy and is completely wrong.

A run:

1. Ingests the fixture corpus into a dedicated Chroma collection, once per run.
2. Puts every case through the real :func:`~app.services.retrieval.answer_question`.
3. Scores each case with deterministic retrieval metrics (no LLM).
4. Optionally scores the batch with deepeval's LLM-as-judge metrics.
5. Writes a result document snapshotting the configuration it ran under.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.models.evaluation import (
    CaseResult,
    EvaluationCase,
    EvaluationRun,
    JudgeMetric,
    RetrievalMetrics,
    RunAggregates,
    RunConfigSnapshot,
    TestSet,
)
from app.models.schemas import ChatResponse
from app.services.chat_memory import ChatMemoryStore
from app.services.chunking import split_pages_into_chunks
from app.services.ingestion import extract_pages_from_pdf
from app.services.llm import AnswerGenerator
from app.services.retrieval import answer_question
from app.services.vector_store import VectorStore


logger = logging.getLogger(__name__)

# The pipeline's own canned no-evidence responses, from answer_question's early
# returns.
REFUSAL_MARKERS = (
    "could not find an answer in the indexed documents",
    "could not find enough supporting evidence",
)

# answer_question's fallback when the LLM call itself fails. A provider failure
# is a third state, neither an answer nor a refusal: counting it as a refusal
# penalised the pipeline for an outage, and counting it as an answer would hide
# one. It is recorded as an errored case and excluded from every rate.
GENERATION_FAILURE_MARKER = "could not generate a final response"


# A real LLM refuses in its own words rather than echoing the canned strings
# above, e.g. "I could not find information about a Platinum plan in the indexed
# documents." Detecting only the canned text scored those as failures.
#
# A refusal needs both halves in the same sentence: an inability-or-absence
# phrase, and a reference to the documents. Requiring the document reference is
# what keeps a legitimate negative *fact* from reading as a refusal — "dental
# treatment is not included under the Basic plan" is an answer, not a decline.
DECLINE_RE = re.compile(
    r"\b(?:"
    r"could ?not|couldn.t|can ?not|cannot|can.t|unable to|not able to|"
    r"no (?:mention|reference|information|details?|record|name)|"
    r"none of the|"
    r"(?:do(?:es)?|did) not (?:mention|contain|specify|include|provide|state|name|list)|"
    r"don.t (?:mention|contain|specify|include|provide|state)|"
    r"(?:is|are|was|were) not (?:mentioned|specified|stated|provided|included|named|listed)"
    r")\b",
    re.IGNORECASE,
)
EVIDENCE_RE = re.compile(
    r"\b(?:document|documents|source|sources|excerpt|excerpts|context|material)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

JUDGE_METRIC_NAMES = (
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
)


def load_testset(path: Path) -> TestSet:
    """Load and validate the test set document at *path*."""
    return TestSet.model_validate_json(path.read_text(encoding="utf-8"))


def is_generation_failure(answer: str) -> bool:
    """Return whether *answer* is the fallback used when the LLM call failed."""
    return GENERATION_FAILURE_MARKER in answer.lower()


def is_refusal(answer: str) -> bool:
    """Return whether *answer* declines to answer for lack of evidence.

    Covers both the pipeline's canned no-evidence responses and a model
    refusing in its own words. The latter is matched per sentence, requiring an
    inability-or-absence phrase *and* a reference to the documents, so that a
    legitimate negative fact ("dental cover is not included on Basic") is not
    mistaken for a refusal.
    """
    lowered = answer.lower()
    if is_generation_failure(answer):
        return False
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return True
    for sentence in SENTENCE_SPLIT_RE.split(answer):
        if DECLINE_RE.search(sentence) and EVIDENCE_RE.search(sentence):
            return True
    return False


def score_retrieval(case: EvaluationCase, response: ChatResponse) -> RetrievalMetrics:
    """Compute the deterministic, judge-free metrics for one case.

    ``first_hit_rank`` is the 1-based position of the first citation whose
    document id is in ``case.expected_sources``; MRR is its reciprocal. For
    negative cases there are no expected sources, so only refusal behaviour is
    meaningful.
    """
    cited_sources = [citation.document_id for citation in response.citations]
    refused = is_refusal(response.answer)

    first_hit_rank: int | None = None
    for rank, document_id in enumerate(cited_sources, start=1):
        if document_id in case.expected_sources:
            first_hit_rank = rank
            break

    return RetrievalMetrics(
        source_hit=first_hit_rank is not None,
        first_hit_rank=first_hit_rank,
        reciprocal_rank=0.0 if first_hit_rank is None else 1.0 / first_hit_rank,
        refused=refused,
        refusal_correct=refused == case.expected_refusal,
        grounded=response.grounded,
        citation_count=len(response.citations),
        cited_sources=cited_sources,
    )


def aggregate(cases: list[CaseResult]) -> RunAggregates:
    """Roll case results up to run level.

    Retrieval rates are computed over *answerable* cases only — a negative case
    correctly citing nothing would otherwise drag ``source_hit_rate`` down for
    behaving exactly as intended.
    """
    answerable = [
        case
        for case in cases
        if case.error is None and case.expected_answer and not _is_negative(case)
    ]
    scored = [case for case in cases if case.error is None]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    judge_means: dict[str, float] = {}
    for name in JUDGE_METRIC_NAMES:
        scores = [
            metric.score
            for case in cases
            for metric in case.judge
            if metric.name == name and metric.status == "scored" and metric.score is not None
        ]
        if scores:
            judge_means[name] = mean(scores)

    return RunAggregates(
        case_count=len(cases),
        errored_cases=sum(1 for case in cases if case.error is not None),
        source_hit_rate=mean([1.0 if case.retrieval.source_hit else 0.0 for case in answerable]),
        mean_reciprocal_rank=mean([case.retrieval.reciprocal_rank for case in answerable]),
        refusal_accuracy=mean([1.0 if case.retrieval.refusal_correct else 0.0 for case in scored]),
        grounded_rate=mean([1.0 if case.retrieval.grounded else 0.0 for case in answerable]),
        mean_latency_seconds=mean([case.latency_seconds for case in scored]),
        judge_means=judge_means,
    )


def _is_negative(case: CaseResult) -> bool:
    """Return whether *case* was a negative (expected-refusal) case."""
    return "negative" in case.tags


class EvaluationRunner:
    """Runs the test set against the real retrieval pipeline."""

    def __init__(self, settings: Settings) -> None:
        """Build a runner bound to an evaluation-scoped copy of *settings*."""
        self._settings = settings.model_copy(
            update={"chroma_collection_name": settings.eval_collection_name}
        )

    def run(self, *, write_result: bool = True) -> EvaluationRun:
        """Execute the full test set and return the result document."""
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        started_at = datetime.now(timezone.utc)
        testset = load_testset(self._settings.eval_testset_path)

        run = EvaluationRun(
            run_id=run_id,
            started_at=started_at,
            status="running",
            testset_version=testset.version,
            config=self._snapshot(),
        )

        try:
            vector_store = VectorStore(self._settings)
            self._ingest_corpus(testset, vector_store)
            run.cases = self._run_cases(testset, vector_store, run_id)
            self._apply_judge(run.cases)
            run.aggregates = aggregate(run.cases)
            run.status = "completed"
        except Exception as exc:
            logger.exception("Evaluation run %s failed.", run_id)
            run.status = "failed"
            run.error = str(exc)
        finally:
            run.finished_at = datetime.now(timezone.utc)

        if write_result:
            self._write_result(run)
        return run

    def _snapshot(self) -> RunConfigSnapshot:
        """Capture the configuration this run executes under."""
        settings = self._settings
        return RunConfigSnapshot(
            retrieval_k=settings.retrieval_k,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_answer_citations=settings.max_answer_citations,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            judge_enabled=settings.eval_judge_enabled,
            judge_model=settings.eval_judge_model if settings.eval_judge_enabled else None,
            eval_threshold=settings.eval_threshold,
        )

    def _ingest_corpus(self, testset: TestSet, vector_store: VectorStore) -> None:
        """Index every corpus PDF under its deterministic document id.

        Called once per run, not per case. Ids are stable, and chunk ids derive
        from them, so a re-ingestion upserts over the previous run's vectors
        instead of accumulating duplicates. Re-ingesting each run also keeps
        results valid for the ``local`` embedding provider regardless of when
        the collection was first built.
        """
        corpus_dir = self._settings.eval_corpus_dir
        for document in testset.corpus:
            path = corpus_dir / document.filename
            if not path.exists():
                raise FileNotFoundError(f"Corpus document missing: {path}")
            pages = extract_pages_from_pdf(path)
            chunks = split_pages_into_chunks(
                pages, self._settings.chunk_size, self._settings.chunk_overlap
            )
            if not chunks:
                raise RuntimeError(f"Corpus document yielded no text: {document.filename}")
            vector_store.add_document(document.document_id, document.filename, chunks)
            logger.info("Indexed %s as %s (%d chunks)", document.filename, document.document_id, len(chunks))

    def _run_cases(
        self, testset: TestSet, vector_store: VectorStore, run_id: str
    ) -> list[CaseResult]:
        """Put every case through the real pipeline and score it deterministically."""
        answer_generator = AnswerGenerator(self._settings)
        chat_memory_store = ChatMemoryStore(self._settings)
        results: list[CaseResult] = []

        for case in testset.cases:
            started = time.perf_counter()
            try:
                response = answer_question(
                    question=case.question,
                    # Per case, never per run: answer_question appends each turn
                    # to chat memory, so a shared session would make
                    # reformulate_query rewrite each query using the previous,
                    # unrelated case — making results order-dependent.
                    session_id=f"eval-{run_id}-{case.id}",
                    settings=self._settings,
                    vector_store=vector_store,
                    answer_generator=answer_generator,
                    chat_memory_store=chat_memory_store,
                    audit=False,
                )
            except Exception as exc:
                logger.exception("Case %s raised.", case.id)
                results.append(
                    CaseResult(
                        case_id=case.id,
                        question=case.question,
                        expected_answer=case.expected_answer,
                        answer="",
                        contexts=[],
                        retrieval=RetrievalMetrics(
                            source_hit=False,
                            refused=False,
                            refusal_correct=False,
                            grounded=False,
                            citation_count=0,
                        ),
                        latency_seconds=round(time.perf_counter() - started, 3),
                        tags=case.tags,
                        error=str(exc),
                    )
                )
                continue

            results.append(
                CaseResult(
                    case_id=case.id,
                    question=case.question,
                    expected_answer=case.expected_answer,
                    answer=response.answer,
                    contexts=[citation.excerpt for citation in response.citations],
                    retrieval=score_retrieval(case, response),
                    latency_seconds=round(time.perf_counter() - started, 3),
                    tags=case.tags,
                    error=(
                        "LLM generation failed; pipeline returned its evidence-only fallback."
                        if is_generation_failure(response.answer)
                        else None
                    ),
                )
            )
        return results

    def _apply_judge(self, cases: list[CaseResult]) -> None:
        """Attach LLM-as-judge metrics to *cases*, in place.

        Marks every metric ``skipped`` with a reason when the judge is disabled
        or unavailable, rather than silently substituting a weaker score — the
        trap the previous per-answer evaluator fell into.
        """
        if not self._settings.eval_judge_enabled:
            self._mark_skipped(cases, "Judge metrics disabled (EVAL_JUDGE_ENABLED=0).")
            return

        try:
            from deepeval import evaluate
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase

            from app.services.eval_judge import build_judge

            judge = build_judge(self._settings)
        except Exception as exc:
            logger.warning("Judge metrics unavailable: %s", exc)
            self._mark_skipped(cases, f"Judge unavailable: {exc}")
            return

        threshold = self._settings.eval_threshold
        # Negative cases are excluded: contextual recall against a refusal is
        # meaningless, and their correctness is already captured by
        # refusal_correct.
        scorable = [
            case for case in cases if case.error is None and case.contexts and not _is_negative(case)
        ]
        scorable_ids = {case.case_id for case in scorable}
        for case in cases:
            if case.case_id not in scorable_ids:
                reason = (
                    "Negative case; correctness captured by refusal_correct."
                    if _is_negative(case)
                    else "No retrieved context to judge."
                )
                case.judge = _skipped_metrics(reason)

        if not scorable:
            return

        test_cases = [
            LLMTestCase(
                input=case.question,
                actual_output=case.answer,
                expected_output=case.expected_answer,
                retrieval_context=case.contexts,
            )
            for case in scorable
        ]
        metrics = [
            AnswerRelevancyMetric(threshold=threshold, model=judge),
            FaithfulnessMetric(threshold=threshold, model=judge),
            ContextualPrecisionMetric(threshold=threshold, model=judge),
            ContextualRecallMetric(threshold=threshold, model=judge),
        ]

        try:
            # One batched call rather than per-case measure(): deepeval
            # parallelises internally, and the judge dominates run time.
            outcome = evaluate(test_cases=test_cases, metrics=metrics)
        except Exception as exc:
            logger.exception("Batched judge evaluation failed.")
            for case in scorable:
                case.judge = _skipped_metrics(f"Judge evaluation failed: {exc}", status="error")
            return

        self._attach_results(scorable, outcome, threshold)

    def _attach_results(self, scorable: list[CaseResult], outcome: object, threshold: float) -> None:
        """Map deepeval's result rows back onto *scorable*, positionally."""
        test_results = getattr(outcome, "test_results", None) or []
        for case, test_result in zip(scorable, test_results, strict=False):
            metrics: list[JudgeMetric] = []
            for data in getattr(test_result, "metrics_data", None) or []:
                name = _normalise_metric_name(getattr(data, "name", "unknown"))
                score = getattr(data, "score", None)
                metrics.append(
                    JudgeMetric(
                        name=name,
                        score=None if score is None else round(float(score), 4),
                        passed=None if score is None else float(score) >= threshold,
                        status="scored" if score is not None else "error",
                        reason=getattr(data, "reason", None) or getattr(data, "error", None),
                    )
                )
            case.judge = metrics or _skipped_metrics("Judge returned no metric data.", status="error")

    def _mark_skipped(self, cases: list[CaseResult], reason: str) -> None:
        """Mark every metric on every case as skipped, with *reason*."""
        for case in cases:
            case.judge = _skipped_metrics(reason)

    def _write_result(self, run: EvaluationRun) -> Path:
        """Persist *run* to ``eval_results_dir`` and return the file path."""
        results_dir = self._settings.eval_results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{run.run_id}.json"
        path.write_text(
            json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Wrote evaluation result to %s", path)
        return path


def _skipped_metrics(reason: str, status: str = "skipped") -> list[JudgeMetric]:
    """Build a full set of non-scored judge metrics carrying *reason*."""
    return [JudgeMetric(name=name, status=status, reason=reason) for name in JUDGE_METRIC_NAMES]


def _normalise_metric_name(name: str) -> str:
    """Map a deepeval metric display name onto our snake_case identifiers."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")
