"""Application configuration loaded from environment variables via Pydantic Settings.

All settings can be overridden with environment variables or a ``backend/.env`` file.
See the project README for a full description of every variable.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DocTongue API"
    api_prefix: str = "/api"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    storage_root: Path = Path(__file__).resolve().parents[2] / "data"
    chat_audit_log_path: Path | None = None
    uploads_dir: Path | None = None
    chroma_dir: Path | None = None
    documents_manifest: Path | None = None
    chroma_collection_name: str = "documents"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    # "local" (hash-based, no key) or "openrouter".
    embedding_provider: str = "local"
    embedding_model: str = "nvidia/nemotron-3.5-embed-8b:free"
    embedding_dimensions: int = 256
    # Send `dimensions` to the provider; disable for models without truncation.
    embedding_send_dimensions: bool = True
    # Send the search_query/search_document `input_type` hint; disable for
    # endpoints that reject the field (OpenAI's own API among them).
    embedding_input_type: bool = True
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None
    # "stub" (no key) or "openrouter".
    llm_provider: str = "stub"
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None
    openai_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 60.0
    max_answer_citations: int = 3
    quality_control_enabled: bool = False
    quality_control_threshold: float = 0.5
    quality_control_fail_open: bool = True
    quality_control_include_reason: bool = True
    chat_memory_window: int = 3
    chat_memory_ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379/0"
    redis_chat_key_prefix: str = "doctongue:chat"

    @field_validator("llm_provider", "embedding_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        """Lower-case and trim provider names so e.g. ``"OpenRouter"`` is accepted."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parents[2] / ".env",
            Path(__file__).resolve().parents[3] / ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def model_post_init(self, __context: object) -> None:
        if self.chat_audit_log_path is None:
            self.chat_audit_log_path = self.storage_root / "chat_audit.txt"
        if self.uploads_dir is None:
            self.uploads_dir = self.storage_root / "uploads"
        if self.chroma_dir is None:
            self.chroma_dir = self.storage_root / "chroma"
        if self.documents_manifest is None:
            self.documents_manifest = self.storage_root / "documents.json"

    def ensure_storage(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.chat_audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        if not self.documents_manifest.exists():
            self.documents_manifest.write_text("[]\n", encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_storage()
    return settings