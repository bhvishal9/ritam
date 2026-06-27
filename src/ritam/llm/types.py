from typing import Protocol

from pydantic import BaseModel

from ritam.cost.types import TokenUsage


class Response(BaseModel):
    response_text: str
    token_usage: TokenUsage
    model_name: str


class LlmClient(Protocol):
    """Protocol describing the interface for LLM clients."""

    def embed_text(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Embed the given text. If embedding_model is provided, use that; otherwise use the client's default"""
        ...

    def generate_response(self, prompt: str, model: str | None = None) -> Response:
        """Generate a response for the given prompt. If model is provided, use that; otherwise use the client's default"""
        ...
