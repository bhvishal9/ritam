import enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_MODEL_NAME = "gemini-embedding-001"
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite-preview"
DEFAULT_QDRANT_CLIENT_URL = "http://localhost:6333"


class VectorStoreType(enum.StrEnum):
    """Vector store types."""

    FILE = "file"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        frozen=True,
    )

    llm_api_key: str
    llm_model: str = Field(
        validation_alias="LLM_MODEL_NAME", default=DEFAULT_MODEL_NAME
    )
    llm_embedding_model: str = Field(
        validation_alias="LLM_EMBEDDING_MODEL_NAME",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
    )
    vector_store: VectorStoreType = Field(
        default=VectorStoreType.QDRANT,
        validation_alias="VECTOR_STORE",
        description="Vector store to use.",
    )
    qdrant_client_url: str = Field(
        default=DEFAULT_QDRANT_CLIENT_URL,
        validation_alias="QDRANT_URL",
        description="Qdrant client URL.",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
