import logging

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from ritam.cost.types import TokenUsage
from ritam.llm.errors import (
    LlmAuthenticationError,
    LlmError,
    LlmInvalidRequestError,
    LlmRateLimitError,
    LlmUnavailableError,
)
from ritam.llm.types import LlmClient, Response

logger = logging.getLogger(__name__)


def _map_gemini_error(err: ClientError | ServerError) -> LlmError:
    """Map Google Gemini ClientError to custom LlmError."""
    error_code = int(err.code)
    if error_code == 400:
        mapped: LlmError = LlmInvalidRequestError(str(err))
    elif error_code in [401, 403]:
        mapped = LlmAuthenticationError(str(err))
    elif error_code == 429:
        mapped = LlmRateLimitError(str(err))
    elif 500 <= error_code < 600:
        mapped = LlmUnavailableError(str(err))
    else:
        mapped = LlmError(str(err))
    logger.warning(
        "upstream_error",
        extra={
            "fields": {
                "provider": "gemini",
                "error_class": mapped.__class__.__name__,
                "upstream_status": error_code,
                "retryable": mapped.retryable,
            }
        },
    )
    return mapped


class GeminiClient(LlmClient):
    """Client for interacting with Google Gemini LLM."""

    def __init__(self, api_key: str, model: str, embedding_model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model

    def embed_text(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Embed the given text. If embedding_model is provided, use that; otherwise use the client's default"""
        try:
            embedding = self.client.models.embed_content(
                model=embedding_model or self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
        except (ClientError, ServerError) as err:
            raise _map_gemini_error(err) from err
        if not embedding.embeddings:
            raise LlmError("Received empty embedding from Gemini")
        embedding_value = embedding.embeddings[0].values
        if embedding_value is None:
            raise LlmError("Received empty embedding from Gemini")
        return embedding_value

    def generate_response(self, prompt: str, model: str | None = None) -> Response:
        """Generate a response for the given prompt. If model is provided, use that; otherwise use the client's default"""
        token_usage = TokenUsage()
        try:
            response = self.client.models.generate_content(
                model=model or self.model,
                contents=prompt,
            )
            response_text = response.text
        except (ClientError, ServerError) as err:
            raise _map_gemini_error(err) from err
        if response_text is None:
            raise LlmError("Received empty response from Gemini")
        if response.usage_metadata is not None:
            usage_metadata = response.usage_metadata
            if (
                usage_metadata.candidates_token_count is not None
                and usage_metadata.prompt_token_count is not None
            ):
                token_usage.output_tokens = usage_metadata.candidates_token_count
                token_usage.input_tokens = usage_metadata.prompt_token_count
        return Response(
            response_text=response_text,
            token_usage=token_usage,
            model_name=model or self.model,
        )
