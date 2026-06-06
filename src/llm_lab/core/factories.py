from urllib.parse import urlparse

from llm_lab.config.settings import VectorStoreType, get_settings
from llm_lab.document_source.local_document_source import LocalDocumentSource
from llm_lab.document_source.types import DocumentSource
from llm_lab.llm.gemini_client import GeminiClient
from llm_lab.llm.types import LlmClient
from llm_lab.vector_store.file.file_store import FileStoreClient
from llm_lab.vector_store.qdrant import QdrantStoreClient
from llm_lab.vector_store.types import VectorStoreClient


def create_llm_client() -> LlmClient:
    settings = get_settings()
    return GeminiClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        embedding_model=settings.llm_embedding_model,
    )


def create_vector_store_client() -> VectorStoreClient:
    settings = get_settings()
    if settings.vector_store == VectorStoreType.FILE:
        return FileStoreClient()
    elif settings.vector_store == VectorStoreType.QDRANT:
        if settings.qdrant_client_url is None:
            raise ValueError("QDRANT_URL is required for the Qdrant vector store")
        return QdrantStoreClient(settings.qdrant_client_url, settings.qdrant_api_key)
    raise NotImplementedError(f"Unsupported vector store type: {settings.vector_store}")


def create_document_source_client() -> DocumentSource:
    settings = get_settings()
    source_uri = settings.source_uri
    parsed = urlparse(source_uri)
    if parsed.scheme == "file":
        return LocalDocumentSource(source_uri)
    raise NotImplementedError(f"Unsupported source uri: {source_uri}")
