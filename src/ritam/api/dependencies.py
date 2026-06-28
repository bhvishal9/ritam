from pydantic import ValidationError

from ritam.api.exceptions import CustomException
from ritam.config.settings import get_settings
from ritam.core.factories import create_llm_client, create_vector_store_client
from ritam.llm.types import LlmClient
from ritam.retrieval.retriever import Retriever
from ritam.vector_store.types import VectorStoreClient


def _configuration_error(err: ValidationError) -> CustomException:
    """Map a settings ValidationError to a config error that names the bad fields.

    Settings are validated as one object inside ``get_settings()``, so a failure
    here is a configuration problem — not a fault of whichever component happened
    to load settings first. Surface the offending field(s) instead of a generic,
    mislabelled message.
    """
    fields = ", ".join(
        ".".join(str(part) for part in error["loc"]) for error in err.errors()
    )
    return CustomException(
        status_code=500,
        message=f"Configuration error: missing or invalid settings: {fields}",
    )


def get_llm_client() -> LlmClient:
    try:
        return create_llm_client()
    except ValidationError as err:
        raise _configuration_error(err) from err


def get_vector_store_client() -> VectorStoreClient:
    try:
        return create_vector_store_client()
    except ValidationError as err:
        raise _configuration_error(err) from err


def get_retriever_client() -> Retriever:
    try:
        similarity_threshold = get_settings().similarity_threshold
    except ValidationError as err:
        raise _configuration_error(err) from err
    return Retriever(get_llm_client(), get_vector_store_client(), similarity_threshold)
