import httpx
import pytest

from app.core.config import Settings
from app.services.llm import (
    AnswerGenerator,
    EmbeddingService,
    _resolve_openrouter_api_key,
)
from app.services.openrouter import (
    INPUT_TYPE_DOCUMENT,
    INPUT_TYPE_QUERY,
    OpenRouterClient,
    OpenRouterError,
)


class _RecordedPost:
    """Captures the arguments of a patched ``httpx.post`` call."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.url: str | None = None
        self.headers: dict[str, str] = {}
        self.json: dict = {}
        self.timeout: float | None = None

    def __call__(self, url, *, headers, json, timeout):
        self.url = url
        self.headers = headers
        self.json = json
        self.timeout = timeout
        return self.response


def _json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


def test_chat_completion_sends_expected_request_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _RecordedPost(
        _json_response({"choices": [{"message": {"content": "  grounded answer  "}}]})
    )
    monkeypatch.setattr(httpx, "post", post)

    client = OpenRouterClient(
        api_key="sk-or-test",
        timeout=12.5,
        site_url="https://example.test",
        app_name="DocTongue",
    )
    answer = client.chat_completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
    )

    assert answer == "grounded answer"
    assert post.url == "https://openrouter.ai/api/v1/chat/completions"
    assert post.headers["Authorization"] == "Bearer sk-or-test"
    assert post.headers["HTTP-Referer"] == "https://example.test"
    assert post.headers["X-Title"] == "DocTongue"
    assert post.timeout == 12.5
    assert post.json == {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.1,
    }


def test_chat_completion_omits_attribution_headers_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _RecordedPost(_json_response({"choices": [{"message": {"content": "ok"}}]}))
    monkeypatch.setattr(httpx, "post", post)

    OpenRouterClient(api_key="sk-or-test").chat_completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert "HTTP-Referer" not in post.headers
    assert "X-Title" not in post.headers
    assert "temperature" not in post.json


def test_chat_completion_strips_trailing_slash_from_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _RecordedPost(_json_response({"choices": [{"message": {"content": "ok"}}]}))
    monkeypatch.setattr(httpx, "post", post)

    OpenRouterClient(
        api_key="sk-or-test", base_url="https://proxy.test/v1/"
    ).chat_completion(model="m", messages=[])

    assert post.url == "https://proxy.test/v1/chat/completions"


def test_chat_completion_requires_api_key() -> None:
    with pytest.raises(OpenRouterError, match="Missing OpenRouter API key"):
        OpenRouterClient(api_key=None).chat_completion(model="m", messages=[])


def test_chat_completion_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "post", _RecordedPost(_json_response({"error": "nope"}, status_code=429))
    )

    with pytest.raises(OpenRouterError, match="429"):
        OpenRouterClient(api_key="sk-or-test").chat_completion(model="m", messages=[])


def test_chat_completion_raises_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "post", _RecordedPost(_json_response({"choices": []})))

    with pytest.raises(OpenRouterError, match="did not contain a message choice"):
        OpenRouterClient(api_key="sk-or-test").chat_completion(model="m", messages=[])


def _embedding_settings(**overrides) -> Settings:
    base = {
        "embedding_provider": "openrouter",
        "embedding_model": "nvidia/nemotron-3.5-embed-8b:free",
        "embedding_dimensions": 2,
        "openrouter_api_key": "sk-or-test",
    }
    base.update(overrides)
    return Settings(**base)


def test_embed_orders_vectors_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    post = _RecordedPost(
        _json_response(
            {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        )
    )
    monkeypatch.setattr(httpx, "post", post)

    vectors = OpenRouterClient(api_key="sk-or-test").embed(
        model="nvidia/nemotron-3.5-embed-8b:free",
        texts=["first", "second"],
        dimensions=2,
        input_type=INPUT_TYPE_DOCUMENT,
    )

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert post.url == "https://openrouter.ai/api/v1/embeddings"
    assert post.json["dimensions"] == 2
    assert post.json["input_type"] == "search_document"


def test_embed_omits_optional_fields_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    post = _RecordedPost(_json_response({"data": [{"index": 0, "embedding": [0.1]}]}))
    monkeypatch.setattr(httpx, "post", post)

    OpenRouterClient(api_key="sk-or-test").embed(model="m", texts=["x"])

    assert "dimensions" not in post.json
    assert "input_type" not in post.json


def test_embed_requires_api_key() -> None:
    with pytest.raises(OpenRouterError, match="Missing OpenRouter API key"):
        OpenRouterClient(api_key=None).embed(model="m", texts=["x"])


def test_embed_rejects_truncated_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "post", _RecordedPost(_json_response({"data": [{"index": 0, "embedding": [0.1]}]}))
    )

    with pytest.raises(OpenRouterError, match="returned 1 embeddings for 2 inputs"):
        OpenRouterClient(api_key="sk-or-test").embed(model="m", texts=["a", "b"])


def test_embedding_service_tags_documents_and_queries_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _RecordedPost(_json_response({"data": [{"index": 0, "embedding": [0.1, 0.2]}]}))
    monkeypatch.setattr(httpx, "post", post)
    service = EmbeddingService(_embedding_settings())

    service.embed_documents(["a document"])
    assert post.json["input_type"] == INPUT_TYPE_DOCUMENT

    assert service.embed_query("a question") == [0.1, 0.2]
    assert post.json["input_type"] == INPUT_TYPE_QUERY
    assert post.json["input"] == ["a question"]


def test_embedding_service_can_disable_input_type_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _RecordedPost(_json_response({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}))
    monkeypatch.setattr(httpx, "post", post)

    EmbeddingService(
        _embedding_settings(embedding_input_type=False, embedding_send_dimensions=False)
    ).embed_documents(["text"])

    assert "input_type" not in post.json
    assert "dimensions" not in post.json


def test_embedding_service_rejects_unexpected_vector_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Model ignored `dimensions` and returned its native 4096-ish width.
    post = _RecordedPost(_json_response({"data": [{"index": 0, "embedding": [0.1] * 4096}]}))
    monkeypatch.setattr(httpx, "post", post)

    with pytest.raises(ValueError, match="returned 4096-dimensional vectors"):
        EmbeddingService(_embedding_settings()).embed_documents(["text"])


def test_embedding_service_local_provider_needs_no_network() -> None:
    service = EmbeddingService(Settings(embedding_provider="local", embedding_dimensions=64))

    documents = service.embed_documents(["first", "second"])
    assert [len(vector) for vector in documents] == [64, 64]
    assert len(service.embed_query("question")) == 64


def test_embed_empty_input_short_circuits() -> None:
    assert EmbeddingService(_embedding_settings()).embed_documents([]) == []


def test_provider_names_are_case_insensitive() -> None:
    settings = Settings(llm_provider="OpenRouter", embedding_provider="  OPENROUTER ")
    assert settings.llm_provider == "openrouter"
    assert settings.embedding_provider == "openrouter"


def test_resolve_openrouter_api_key_prefers_explicit_override() -> None:
    settings = Settings(openrouter_api_key="from-openrouter-var")
    assert _resolve_openrouter_api_key(settings) == "from-openrouter-var"

    settings = Settings(openrouter_api_key="from-openrouter-var", llm_api_key="override")
    assert _resolve_openrouter_api_key(settings) == "override"


def test_answer_generator_rejects_unknown_provider() -> None:
    generator = AnswerGenerator(Settings(llm_provider="litellm"))
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        generator.generate_answer("q", ["context"])


def test_embedding_service_rejects_unknown_provider() -> None:
    service = EmbeddingService(Settings(embedding_provider="gemini"))
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        service.embed_documents(["text"])


def test_answer_generator_uses_openrouter_for_grounded_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = _RecordedPost(
        _json_response({"choices": [{"message": {"content": "Answer [1]"}}]})
    )
    monkeypatch.setattr(httpx, "post", post)

    settings = Settings(
        llm_provider="openrouter",
        llm_model="openai/gpt-4o-mini",
        openrouter_api_key="sk-or-test",
    )
    answer = AnswerGenerator(settings).generate_answer("What is it?", ["Evidence."])

    assert answer == "Answer [1]"
    assert post.json["model"] == "openai/gpt-4o-mini"
    assert "Source 1:\nEvidence." in post.json["messages"][1]["content"]


def test_reformulate_query_returns_question_when_history_empty() -> None:
    settings = Settings(llm_provider="openrouter", openrouter_api_key="sk-or-test")
    assert AnswerGenerator(settings).reformulate_query("standalone?", []) == "standalone?"
