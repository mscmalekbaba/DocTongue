"""FastAPI dependency providers for all singleton application services.

Each getter is decorated with :func:`functools.lru_cache` so only one
instance is constructed per worker process lifetime.
"""

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.document_store import DocumentStore
from app.services.chat_memory import ChatMemoryStore
from app.services.llm import AnswerGenerator, EmbeddingService
from app.services.vector_store import VectorStore


def settings_dependency() -> Settings:
    """FastAPI dependency that returns the cached application settings."""
    return get_settings()


@lru_cache
def get_document_store() -> DocumentStore:
    """Return the singleton :class:`DocumentStore` instance."""
    return DocumentStore(get_settings())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return the singleton :class:`EmbeddingService` instance."""
    return EmbeddingService(get_settings())


@lru_cache
def get_answer_generator() -> AnswerGenerator:
    """Return the singleton :class:`AnswerGenerator` instance."""
    return AnswerGenerator(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    """Return the singleton :class:`VectorStore` instance."""
    return VectorStore(get_settings())


@lru_cache
def get_chat_memory_store() -> ChatMemoryStore:
    """Return the singleton :class:`ChatMemoryStore` instance."""
    return ChatMemoryStore(get_settings())
