from pydantic import ValidationError

from ritam.api.exceptions import CustomException
from ritam.core.factories import create_llm_client, create_vector_store_client
from ritam.llm.types import LlmClient
from ritam.retrieval.retriever import Retriever
from ritam.vector_store.types import VectorStoreClient


def get_llm_client() -> LlmClient:
    try:
        return create_llm_client()
    except ValidationError as err:
        raise CustomException(
            status_code=500,
            message="LLM configuration error: missing or invalid environment variables",
        ) from err


def get_vector_store_client() -> VectorStoreClient:
    try:
        return create_vector_store_client()
    except ValidationError as err:
        raise CustomException(
            status_code=500,
            message="Vector Store configuration error: missing or invalid environment variables",
        ) from err


def get_retriever_client() -> Retriever:
    return Retriever(get_llm_client(), get_vector_store_client())
