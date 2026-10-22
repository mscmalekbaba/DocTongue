from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.dependencies import (
    get_answer_generator,
    get_document_store,
    get_embedding_service,
    get_quality_evaluator,
    get_vector_store,
)
from app.main import create_app


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every storage path at a per-test temporary directory.

    This is autouse because tests that build a bare ``Settings()`` rather than
    going through the ``client`` fixture would otherwise resolve the real
    ``backend/data`` directory and write into it — the chat audit log in
    particular, which is a tracked file. Redirecting via the environment covers
    both paths, since ``Settings`` reads these names itself.

    Each entry is set explicitly rather than relying on ``STORAGE_ROOT`` alone,
    because the derived defaults only apply when the specific variable is
    absent and a developer's ``.env`` may set any of them.
    """
    storage_root = tmp_path / "data"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    monkeypatch.setenv("UPLOADS_DIR", str(storage_root / "uploads"))
    monkeypatch.setenv("CHROMA_DIR", str(storage_root / "chroma"))
    monkeypatch.setenv("DOCUMENTS_MANIFEST", str(storage_root / "documents.json"))
    monkeypatch.setenv("CHAT_AUDIT_LOG_PATH", str(storage_root / "chat_audit.txt"))


@pytest.fixture(autouse=True)
def zero_key_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the keyless providers so no test can make a live API call.

    A developer ``.env`` configured for OpenRouter would otherwise be picked up
    by any test constructing settings directly.
    """
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Return a ``TestClient`` with all cached service singletons reset."""
    _clear_service_caches()

    app = create_app()

    with TestClient(app) as test_client:
        yield test_client

    _clear_service_caches()


def _clear_service_caches() -> None:
    """Drop every ``lru_cache``d provider so the next call re-reads settings."""
    get_settings.cache_clear()
    get_document_store.cache_clear()
    get_embedding_service.cache_clear()
    get_answer_generator.cache_clear()
    get_quality_evaluator.cache_clear()
    get_vector_store.cache_clear()
