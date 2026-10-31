"""OpenRouter-backed judge model for deepeval metrics.

deepeval defaults to calling OpenAI directly, which fails without
``OPENAI_API_KEY`` even when the application itself is configured for
OpenRouter.  This adapter routes judge calls through the same
:class:`~app.services.openrouter.OpenRouterClient` the rest of the application
uses, so one key and one provider cover both answering and evaluation.

Judge calls run at ``temperature=0`` and cap their own retries: the judge is
roughly 85% of a run's token spend, and an uncapped retry loop on malformed
JSON is the one way an evaluation run becomes expensive.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import Settings
from app.services.llm import resolve_openrouter_api_key
from app.services.openrouter import OpenRouterClient


logger = logging.getLogger(__name__)

JSON_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else. "
    "Do not wrap it in markdown fences or add commentary."
)


class JudgeUnavailableError(RuntimeError):
    """Raised when the judge cannot be constructed, e.g. a missing API key."""


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, which models add unbidden."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()


def _extract_json_object(text: str) -> str:
    """Return the outermost JSON object found in *text*.

    Falls back to the original string when no balanced object is present, so
    the caller's parse error carries the real payload.
    """
    candidate = _strip_code_fence(text)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return candidate
    return candidate[start : end + 1]


def build_judge(settings: Settings) -> Any:
    """Construct the deepeval judge model for *settings*.

    Imported lazily and returned as :class:`~typing.Any` so that neither
    deepeval nor this module is required to import the application.

    Raises:
        JudgeUnavailableError: If deepeval is not installed or no API key is
            configured for the judge.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise JudgeUnavailableError(f"deepeval is not installed: {exc}") from exc

    api_key = resolve_openrouter_api_key(settings)
    if not api_key:
        raise JudgeUnavailableError(
            "No OpenRouter API key configured; set OPENROUTER_API_KEY to enable judge metrics."
        )

    class OpenRouterJudge(DeepEvalBaseLLM):
        """deepeval judge model that delegates to OpenRouter."""

        def __init__(self, judge_settings: Settings) -> None:
            self._settings = judge_settings
            self._client = OpenRouterClient(
                api_key=resolve_openrouter_api_key(judge_settings),
                base_url=judge_settings.openrouter_base_url,
                timeout=judge_settings.llm_timeout_seconds,
                site_url=judge_settings.openrouter_site_url,
                app_name=judge_settings.openrouter_app_name or judge_settings.app_name,
            )
            super().__init__(model_name=judge_settings.eval_judge_model)

        def get_model_name(self) -> str:
            return self._settings.eval_judge_model

        def load_model(self) -> Any:
            """Return the underlying client; deepeval requires the hook to exist."""
            return self._client

        def generate(self, prompt: str, schema: Any | None = None, **_: Any) -> Any:
            """Complete *prompt*, validating against *schema* when deepeval supplies one.

            Retries are bounded by ``eval_judge_max_retries``. A schema is only
            ever retried on a parse/validation failure, never on a transport
            error, which is left to propagate.
            """
            attempts = max(1, self._settings.eval_judge_max_retries + 1)
            last_error: Exception | None = None
            for attempt in range(attempts):
                content = self._complete(prompt, expect_json=schema is not None)
                if schema is None:
                    return content
                payload = _extract_json_object(content)
                try:
                    return schema.model_validate_json(payload)
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Judge returned unparseable JSON (attempt %d/%d): %s",
                        attempt + 1,
                        attempts,
                        exc,
                    )
            raise ValueError(
                f"Judge did not return schema-valid JSON after {attempts} attempts: {last_error}"
            )

        async def a_generate(self, prompt: str, schema: Any | None = None, **kwargs: Any) -> Any:
            """Async entry point. The underlying client is synchronous.

            deepeval drives metrics concurrently via ``asyncio``; this keeps the
            call correct, at the cost of not overlapping HTTP waits.
            """
            return self.generate(prompt, schema, **kwargs)

        def _complete(self, prompt: str, expect_json: bool) -> str:
            messages = [{"role": "user", "content": prompt}]
            if expect_json:
                messages.insert(0, {"role": "system", "content": JSON_INSTRUCTION})
            return self._client.chat_completion(
                model=self._settings.eval_judge_model,
                messages=messages,
                temperature=0.0,
            )

    return OpenRouterJudge(settings)


def judge_json_loads(content: str) -> dict[str, Any]:
    """Parse a judge response as a JSON object. Exposed for tests."""
    return json.loads(_extract_json_object(content))
