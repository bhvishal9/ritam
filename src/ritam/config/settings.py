import enum
from functools import lru_cache
from typing import Self
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreType(enum.StrEnum):
    """Vector store types."""

    FILE = "file"
    QDRANT = "qdrant"


def _is_local(url: str) -> bool:
    """Return True if the URL points at a loopback host."""
    return urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True,
    )

    llm_api_key: str = Field(
        description="LLM API key",
    )
    llm_embedding_model: str = Field(
        validation_alias="LLM_EMBEDDING_MODEL_NAME",
        default="gemini-embedding-001",
        description="Embedding model name.",
    )
    llm_model: str = Field(
        validation_alias="LLM_MODEL_NAME",
        default="gemini-3.1-flash-lite",
        description="LLM model name.",
    )
    source_uri: str | None = Field(
        default=None,
        description="Source URI where documents are stored.",
    )
    vector_store: VectorStoreType = Field(
        default=VectorStoreType.QDRANT,
        validation_alias="VECTOR_STORE",
        description="Vector store to use.",
    )
    qdrant_client_url: str | None = Field(
        validation_alias="QDRANT_URL",
        description="Qdrant client URL.",
        default=None,
    )
    qdrant_api_key: str | None = Field(
        description="Qdrant API key",
        default=None,
    )

    @model_validator(mode="after")
    def validate_qdrant_settings(self) -> Self:
        """Qdrant needs a URL; a non-local (cloud) URL also needs an API key."""
        if self.vector_store != VectorStoreType.QDRANT:
            return self
        if not self.qdrant_client_url:
            raise ValueError("QDRANT_URL must be provided when VECTOR_STORE=qdrant.")
        if not _is_local(self.qdrant_client_url) and not self.qdrant_api_key:
            raise ValueError(
                "QDRANT_API_KEY must be provided when using a non-local Qdrant URL."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
