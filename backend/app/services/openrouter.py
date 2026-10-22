"""HTTP client for the OpenRouter API.

OpenRouter exposes OpenAI-compatible ``/chat/completions`` and ``/embeddings``
endpoints that proxy many upstream models behind a single API key, so
:class:`OpenRouterClient` covers both the generation and the hosted embedding
path.  Because the API is OpenAI-shaped, the same client also works against
OpenAI itself or any other compatible service by overriding ``base_url``.

Embeddings can alternatively be produced locally by the hash-based fallback in
:mod:`app.services.llm`, which needs no API key.
"""

from typing import Any

import httpx


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: ``input_type`` hints that tell asymmetric embedding models (such as the
#: NVIDIA Nemotron embed family) whether a text is a search query or an
#: indexed document.  Embedding both sides with the same hint measurably
#: degrades retrieval on these models.
INPUT_TYPE_QUERY = "search_query"
INPUT_TYPE_DOCUMENT = "search_document"


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails or returns no usable content."""


class OpenRouterClient:
    """Minimal synchronous client for the OpenRouter chat and embeddings APIs.

    Args:
        api_key: OpenRouter API key (``sk-or-...``).
        base_url: API root; defaults to :data:`DEFAULT_OPENROUTER_BASE_URL`.
        timeout: Per-request timeout in seconds.
        site_url: Optional value for the ``HTTP-Referer`` header, used by
            OpenRouter for app attribution on its public leaderboards.
        app_name: Optional value for the ``X-Title`` attribution header.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout: float = 60.0,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        """Initialise the client; credentials are validated lazily on first request."""
        self._api_key = api_key
        self._base_url = (base_url or DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._site_url = site_url
        self._app_name = app_name

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Request a chat completion and return the content of the first choice.

        Args:
            model: OpenRouter model slug, e.g. ``"openai/gpt-4o-mini"``.
            messages: OpenAI-style chat messages.
            temperature: Optional sampling temperature.
            max_tokens: Optional cap on generated tokens.

        Returns:
            The assistant message content, stripped of surrounding whitespace.

        Raises:
            OpenRouterError: If the API key is missing, the request fails, the
                response is not valid JSON, or no message content is returned.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = self._post("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(
                f"OpenRouter response did not contain a message choice: {data!r}"
            ) from exc
        if not content:
            raise OpenRouterError("OpenRouter returned an empty message content.")
        return content.strip()

    def embed(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None = None,
        input_type: str | None = None,
    ) -> list[list[float]]:
        """Return one embedding vector per entry in *texts*, in input order.

        Args:
            model: Embedding model slug, e.g.
                ``"nvidia/nemotron-3.5-embed-8b:free"``.
            texts: Strings to embed.
            dimensions: Optional output width.  Models that support truncation
                return a vector sliced to this size (re-normalised); models
                that do not may ignore it, so the caller should verify the
                width it gets back.
            input_type: Optional asymmetry hint, normally
                :data:`INPUT_TYPE_QUERY` or :data:`INPUT_TYPE_DOCUMENT`.

        Returns:
            Embedding vectors ordered to match *texts*.

        Raises:
            OpenRouterError: If the API key is missing, the request fails, or
                the response payload is malformed.
        """
        payload: dict[str, Any] = {"model": model, "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if input_type is not None:
            payload["input_type"] = input_type

        data = self._post("/embeddings", payload)
        try:
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise OpenRouterError(
                f"OpenRouter response did not contain embedding data: {data!r}"
            ) from exc
        if len(vectors) != len(texts):
            # Callers pair vectors with inputs positionally, so a short response
            # would silently misalign chunks with their embeddings.
            raise OpenRouterError(
                f"OpenRouter returned {len(vectors)} embeddings for {len(texts)} inputs."
            )
        return vectors

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST *payload* to *path* and return the decoded JSON body."""
        if not self._api_key:
            raise OpenRouterError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY (or LLM_API_KEY)."
            )
        try:
            response = httpx.post(
                f"{self._base_url}{path}",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise OpenRouterError(
                f"OpenRouter request to {path} failed with "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter request to {path} failed: {exc}") from exc
        except ValueError as exc:
            raise OpenRouterError("OpenRouter returned a non-JSON response.") from exc

    def _headers(self) -> dict[str, str]:
        """Build request headers, including optional OpenRouter attribution headers."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        return headers
