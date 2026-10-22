import os

import pytest

from app.core.config import Settings
from app.services.llm import AnswerGenerator, EmbeddingService


def test_live_openrouter_smoke() -> None:
    if os.getenv("RUN_LIVE_LLM_SMOKE_TEST") != "1":
        pytest.skip("Set RUN_LIVE_LLM_SMOKE_TEST=1 to run the live OpenRouter smoke test.")

    settings = Settings()
    if settings.llm_provider != "openrouter":
        pytest.skip("Set LLM_PROVIDER=openrouter before running the live smoke test.")

    answer = AnswerGenerator(settings).generate_answer(
        question="How does DocTongue answer questions?",
        contexts=["DocTongue uses retrieved document evidence to answer questions."],
    )
    assert answer.strip()

    if settings.embedding_provider != "openrouter":
        pytest.skip(
            "Set EMBEDDING_PROVIDER=openrouter to also smoke-test hosted embeddings."
        )

    service = EmbeddingService(settings)
    document = "DocTongue uses retrieved document evidence to answer questions."
    documents = service.embed_documents([document])
    assert len(documents) == 1
    assert documents[0]

    # Asymmetric models return a different vector for the same text embedded as
    # a query, so this also proves the input_type hint reached the provider.
    query = service.embed_query(document)
    assert len(query) == len(documents[0])
