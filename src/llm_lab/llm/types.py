from typing import Protocol


class LlmClient(Protocol):
    """Protocol describing the interface for LLM clients."""

    def embed_text(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Embed the given text. If embedding_model is provided, use that; otherwise use the client's default"""
        ...

    def generate_response(self, prompt: str, model: str | None = None) -> str:
        """Generate a response for the given prompt. If model is provided, use that; otherwise use the client's default"""
        ...
