import enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        default="gemini-3.1-flash-lite-preview",
        description="LLM model name.",
    )
    qdrant_client_url: str = Field(
        default="http://localhost:6333",
        validation_alias="QDRANT_URL",
        description="Qdrant client URL.",
    )
    source_uri: str = Field(
        description="Source URI where documents are stored.",
    )
    vector_store: VectorStoreType = Field(
        default=VectorStoreType.QDRANT,
        validation_alias="VECTOR_STORE",
        description="Vector store to use.",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
