import time

from llm_lab.config.variables import (
    CANDIDATE_MULTIPLIER,
    MAX_CANDIDATES,
    SIMILARITY_SCORE_THRESHOLD,
)
from llm_lab.llm.types import LlmClient
from llm_lab.observability.context import (
    candidate_k_context_var,
    chunks_return_context_var,
    embed_ms_context_var,
    retrieve_ms_context_var,
)
from llm_lab.vector_store.types import ScoredChunk, VectorStoreClient


class Retriever:
    """Class for scoring chunks based on cosine similarity."""

    def __init__(
        self,
        llm_client: LlmClient,
        vector_store_client: VectorStoreClient,
    ) -> None:
        self.llm_client = llm_client
        self.vector_store_client = vector_store_client

    def search(
        self, dataset: str, embedding_model: str, query: str, top_k: int
    ) -> list[ScoredChunk]:
        candidate_k = min(top_k * CANDIDATE_MULTIPLIER, MAX_CANDIDATES)
        candidate_k_context_var.set(candidate_k)
        embedding_start_time = time.perf_counter()
        query_embedding = self.llm_client.embed_text(query, embedding_model)
        embedding_time = round((time.perf_counter() - embedding_start_time) * 1000, 3)
        embed_ms_context_var.set(embedding_time)
        retrieve_start_time = time.perf_counter()
        scored_chunks = self.vector_store_client.query(
            dataset, embedding_model, query_embedding, candidate_k
        )
        retrieve_time = round((time.perf_counter() - retrieve_start_time) * 1000, 3)
        retrieve_ms_context_var.set(retrieve_time)
        selected_chunks = [
            sc for sc in scored_chunks if sc.score >= SIMILARITY_SCORE_THRESHOLD
        ][:top_k]
        chunks_return_context_var.set(len(selected_chunks))
        return selected_chunks
