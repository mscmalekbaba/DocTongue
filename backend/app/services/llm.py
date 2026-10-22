"""LLM and embedding service wrappers.

Provides two main classes:

- :class:`EmbeddingService`: converts text into embedding vectors via the
  OpenRouter embeddings endpoint, or a local hash-based fallback when
  ``embedding_provider="local"``.
- :class:`AnswerGenerator`: generates grounded answers and reformulates
  multi-turn queries via OpenRouter, or a zero-key stub for local development.

Private helpers handle API key resolution and the hash-based local embedding
strategy.
"""

from collections import Counter
import math
import re

from app.core.config import Settings
from app.services.openrouter import (
    INPUT_TYPE_DOCUMENT,
    INPUT_TYPE_QUERY,
    OpenRouterClient,
)


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

ANSWER_SYSTEM_PROMPT = (
    "Answer only from the provided document excerpts. "
    "Cite every factual claim with source markers like [1] or [2], "
    "where numbers map to the provided Source blocks. "
    "If the answer is not supported, say you could not find it in the indexed documents."
)
REFORMULATION_SYSTEM_PROMPT = (
    "Rewrite the latest user question into a standalone retrieval query. "
    "Use conversation history, including assistant answers, only to resolve references. "
    "Keep it concise and return only the rewritten query text."
)


class EmbeddingService:
    """Converts text into dense embedding vectors.

    When ``embedding_provider`` is set to ``"local"`` a deterministic
    hash-based bag-of-words vector is returned (no API key required).
    With ``"openrouter"`` the texts are sent to OpenRouter's OpenAI-compatible
    ``/embeddings`` endpoint, which can also be pointed at OpenAI or another
    compatible service via ``embedding_api_base``.

    Queries and documents are embedded through separate methods because
    asymmetric models (the NVIDIA Nemotron embed family among them) expect an
    ``input_type`` hint saying which side of the comparison a text is on.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the service with application *settings*."""
        self._settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* as indexed documents, for storage in the vector DB."""
        return self._embed(texts, input_type=INPUT_TYPE_DOCUMENT)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search *text* as a retrieval query."""
        vectors = self._embed([text], input_type=INPUT_TYPE_QUERY)
        return vectors[0]

    def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        """Return one embedding vector per entry in *texts*.

        Args:
            texts: Strings to embed; an empty list short-circuits.
            input_type: Asymmetry hint forwarded to the provider when
                ``embedding_input_type`` is enabled.

        Returns:
            List of float vectors, ordered to match *texts*.

        Raises:
            ValueError: If the configured ``embedding_provider`` is not
                supported, or the provider returned vectors of an unexpected
                width.
            OpenRouterError: If the hosted embeddings request fails.
        """
        if not texts:
            return []
        if self._settings.embedding_provider == "local":
            return [
                _hashed_embedding(text, self._settings.embedding_dimensions)
                for text in texts
            ]
        if self._settings.embedding_provider != "openrouter":
            raise ValueError(
                f"Unsupported embedding provider: {self._settings.embedding_provider}. "
                'Expected "local" or "openrouter".'
            )

        client = OpenRouterClient(
            api_key=(
                self._settings.embedding_api_key
                or _resolve_openrouter_api_key(self._settings)
            ),
            base_url=self._settings.embedding_api_base or self._settings.openrouter_base_url,
            timeout=self._settings.llm_timeout_seconds,
            site_url=self._settings.openrouter_site_url,
            app_name=self._settings.openrouter_app_name or self._settings.app_name,
        )
        requested_dimensions = (
            self._settings.embedding_dimensions
            if self._settings.embedding_send_dimensions
            else None
        )
        vectors = client.embed(
            model=self._settings.embedding_model,
            texts=texts,
            dimensions=requested_dimensions,
            input_type=input_type if self._settings.embedding_input_type else None,
        )
        _verify_embedding_width(vectors, requested_dimensions, self._settings.embedding_model)
        return vectors


class AnswerGenerator:
    """Generates grounded answers and reformulates queries using an LLM.

    Supports the built-in ``"stub"`` provider for zero-key local development
    and ``"openrouter"`` for production use.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the generator with application *settings*."""
        self._settings = settings

    def generate_answer(self, question: str, contexts: list[str]) -> str:
        """Generate an answer to *question* grounded in the provided *contexts*.

        The LLM is instructed to cite every factual claim with ``[N]`` markers
        that correspond to the source blocks passed in *contexts*.
        Falls back to :func:`_fallback_answer` when ``llm_provider="stub"``.

        Args:
            question: The user's original question.
            contexts: Ordered list of document excerpts to cite from.

        Returns:
            Answer string that may contain ``[N]`` citation markers.

        Raises:
            ValueError: If the configured ``llm_provider`` is not supported.
            OpenRouterError: If the OpenRouter request fails.
        """
        if self._settings.llm_provider == "stub":
            return _fallback_answer(question, contexts)
        self._require_supported_provider()

        prompt_context = "\n\n".join(
            f"Source {index + 1}:\n{context}"
            for index, context in enumerate(contexts)
        )
        return self._client().chat_completion(
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{prompt_context}",
                },
            ],
        )

    def reformulate_query(self, question: str, recent_turns: list[dict[str, str]]) -> str:
        """Rewrite *question* into a standalone retrieval query using conversation history.

        Resolves pronouns and implicit references in *question* by consulting
        *recent_turns* (most-recent first).  Returns *question* unchanged when
        the history is empty.

        Args:
            question: The latest user question.
            recent_turns: Recent chat turns, each a dict with ``"question"``
                and ``"answer"`` keys, ordered most-recent first.

        Returns:
            A self-contained query string suitable for vector search.

        Raises:
            ValueError: If the configured ``llm_provider`` is not supported.
            OpenRouterError: If the OpenRouter request fails.
        """
        if not recent_turns:
            return question

        ordered_history = list(reversed(recent_turns))
        if self._settings.llm_provider == "stub":
            history_parts = [
                f"Q: {turn.get('question', '')} A: {turn.get('answer', '')}".strip()
                for turn in ordered_history
            ]
            history = " ".join(part for part in history_parts if part)
            return f"{history} {question}".strip()
        self._require_supported_provider()

        history_lines: list[str] = []
        for index, turn in enumerate(ordered_history):
            history_lines.append(
                f"Turn {index + 1} User: {turn.get('question', '').strip()}"
            )
            history_lines.append(
                f"Turn {index + 1} Assistant: {turn.get('answer', '').strip()}"
            )
        history_block = "\n".join(history_lines)

        rewritten = self._client().chat_completion(
            model=self._settings.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": REFORMULATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Recent conversation turns (oldest to newest):\n"
                        f"{history_block}\n\n"
                        f"Current user question: {question}"
                    ),
                },
            ],
        )
        return rewritten or question

    def _client(self) -> OpenRouterClient:
        """Build an :class:`OpenRouterClient` from the current settings."""
        return OpenRouterClient(
            api_key=_resolve_openrouter_api_key(self._settings),
            base_url=self._settings.openrouter_base_url,
            timeout=self._settings.llm_timeout_seconds,
            site_url=self._settings.openrouter_site_url,
            app_name=self._settings.openrouter_app_name or self._settings.app_name,
        )

    def _require_supported_provider(self) -> None:
        """Raise :class:`ValueError` unless ``llm_provider`` is ``"openrouter"``."""
        if self._settings.llm_provider != "openrouter":
            raise ValueError(
                f"Unsupported LLM provider: {self._settings.llm_provider}. "
                'Expected "openrouter" or "stub".'
            )


def _resolve_openrouter_api_key(settings: Settings) -> str | None:
    """Return the OpenRouter key, preferring an explicit ``llm_api_key`` override."""
    return settings.llm_api_key or settings.openrouter_api_key


def _verify_embedding_width(
    vectors: list[list[float]],
    requested_dimensions: int | None,
    model: str,
) -> None:
    """Raise :class:`ValueError` if *vectors* are not the width that was requested.

    Not every embedding model honours ``dimensions``; a model that silently
    ignores it returns its native width instead.  Catching that here gives a
    clear error rather than a confusing dimension mismatch from the vector DB
    on the next write.
    """
    if requested_dimensions is None or not vectors:
        return
    actual = len(vectors[0])
    if actual != requested_dimensions:
        raise ValueError(
            f"Embedding model {model} returned {actual}-dimensional vectors but "
            f"EMBEDDING_DIMENSIONS is {requested_dimensions}. Either set "
            f"EMBEDDING_DIMENSIONS={actual}, or set EMBEDDING_SEND_DIMENSIONS=0 "
            "if this model does not support truncation."
        )


def _hashed_embedding(text: str, dimensions: int) -> list[float]:
    """Produce a normalised hash-based bag-of-words embedding for *text*.

    Tokens are mapped into a *dimensions*-dimensional vector by hashing;
    counts are accumulated per bucket, then L2-normalised.
    """
    counts = Counter(token.lower() for token in TOKEN_RE.findall(text))
    vector = [0.0] * dimensions
    for token, count in counts.items():
        index = hash(token) % dimensions
        vector[index] += float(count)
    return _normalize_vector(vector)


def _normalize_vector(vector: list[float]) -> list[float]:
    """Return *vector* scaled to unit length; return it unchanged if its norm is zero."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _fallback_answer(question: str, contexts: list[str]) -> str:
    """Construct a simple template answer from *contexts* when no LLM is available."""
    if not contexts:
        return "I could not find an answer in the indexed documents."
    lead = contexts[0]
    if len(contexts) == 1:
        return f"Based on the indexed documents: {lead} [1]"
    supporting = contexts[1]
    return (
        "Based on the indexed documents, the most relevant evidence is: "
        f"{lead} [1] Additional supporting context: {supporting} [2]"
    )


def lexical_overlap_score(question: str, text: str) -> float:
    """Return the fraction of *question* tokens that also appear in *text*.

    Used as a lightweight grounding check and quality-control fallback.
    Returns ``0.0`` when either string contains no alphanumeric tokens.
    """
    question_terms = {token.lower() for token in TOKEN_RE.findall(question)}
    context_terms = {token.lower() for token in TOKEN_RE.findall(text)}
    if not question_terms or not context_terms:
        return 0.0
    return len(question_terms & context_terms) / len(question_terms)
