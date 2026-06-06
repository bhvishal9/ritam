import logging
import time

from pydantic import BaseModel

from ritam.llm.types import LlmClient
from ritam.observability.context import generate_ms_context_var, stage
from ritam.retrieval.retriever import Retriever
from ritam.vector_store.types import ScoredChunk

logger = logging.getLogger(__name__)


def build_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    """Build a prompt for the LLM based on the question and chunks."""
    context_parts = []
    for sc in chunks:
        chunk = sc.indexed_chunk
        context_parts.append(
            f"Source: {chunk.source} (chunk {chunk.chunk_id})\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


class QueryResult(BaseModel):
    answer: str
    chunks: list[ScoredChunk]


class RagService:
    def __init__(
        self,
        llm_client: LlmClient,
        retriever: Retriever,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever

    def answer_question(
        self,
        dataset: str,
        embedding_model: str,
        query: str,
        top_k: int,
    ) -> QueryResult:
        """Answer a question using a simple RAG pipeline."""

        top_chunks = self.retriever.search(dataset, embedding_model, query, top_k)
        if not top_chunks:
            logger.info("empty_retrieval", extra={"fields": {"abstained": True}})
            return QueryResult(
                answer="No relevant information found to answer the question.",
                chunks=top_chunks,
            )
        prompt = build_prompt(query, top_chunks)
        with stage("generate"):
            start_time = time.perf_counter()
            response = self.llm_client.generate_response(prompt)
            generate_ms = round(((time.perf_counter() - start_time) * 1000), 3)
            generate_ms_context_var.set(generate_ms)
        return QueryResult(
            answer=response,
            chunks=top_chunks,
        )
