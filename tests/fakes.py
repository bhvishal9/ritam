from ritam.vector_store.types import IndexedChunk, ScoredChunk, VectorStoreClient


class FakeLlmClient:
    def __init__(
        self,
        response: str | None = None,
        generate_error: Exception | None = None,
    ) -> None:
        self._response = response
        self._generate_error = generate_error

    def embed_text(self, text: str, embedding_model: str | None = None) -> list[float]:
        return [0.1, 0.2, 0.3]

    def generate_response(self, prompt: str, model: str | None = None) -> str:
        if self._generate_error is not None:
            raise self._generate_error
        if self._response is None:
            raise NotImplementedError
        return self._response


class NoCallLlmClient:
    def embed_text(self, text: str, embedding_model: str | None = None) -> list[float]:
        return [1.0, 0.0]

    def generate_response(self, prompt: str, model: str | None = None) -> str:
        raise AssertionError(
            "generate_response should not be called when no chunks are returned."
        )


class FakeRetriever:
    """Fake Retriever that returns configurable chunks or raises on search."""

    def __init__(
        self,
        chunks: list[ScoredChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._error = error

    def search(
        self, dataset: str, embedding_model: str, query: str, top_k: int
    ) -> list[ScoredChunk]:
        if self._error is not None:
            raise self._error
        return self._chunks[:top_k]


class FakeVectorStoreClient(VectorStoreClient):
    """Fake VectorStoreClient that returns a configurable list of ScoredChunks."""

    def __init__(self, scored_chunks: list[ScoredChunk] | None = None) -> None:
        self._scored_chunks = scored_chunks or []

    def store(
        self,
        indexed_chunks: list[IndexedChunk],
        dataset: str,
        embedding_model: str,
        docs_count: int,
    ) -> None:
        pass

    def query(
        self,
        dataset: str,
        embedding_model: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[ScoredChunk]:
        return self._scored_chunks
