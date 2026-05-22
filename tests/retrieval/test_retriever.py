from llm_lab.retrieval.retriever import Retriever
from llm_lab.vector_store.types import IndexedChunk, ScoredChunk
from tests.fakes import FakeLlmClient, FakeVectorStoreClient


class TestRetriever:
    def test_search_filters_by_threshold_and_returns_top_k(
        self, fake_llm_client: FakeLlmClient
    ) -> None:
        chunk_a = IndexedChunk(
            text="high similarity",
            doc_path="a.md",
            source="a.md",
            embedding=[1.0, 0.0],
            chunk_id=0,
        )
        chunk_b = IndexedChunk(
            text="medium similarity",
            doc_path="b.md",
            source="b.md",
            embedding=[1.0, 1.0],
            chunk_id=1,
        )
        chunk_c = IndexedChunk(
            text="low similarity",
            doc_path="c.md",
            source="c.md",
            embedding=[0.0, 1.0],
            chunk_id=2,
        )
        # Scores: a=1.0 (pass), b=0.707 (pass, just above 0.70 threshold), c=0.0 (fail)
        scored_chunks = [
            ScoredChunk(score=1.0, indexed_chunk=chunk_a),
            ScoredChunk(score=0.707, indexed_chunk=chunk_b),
            ScoredChunk(score=0.0, indexed_chunk=chunk_c),
        ]

        retriever = Retriever(fake_llm_client, FakeVectorStoreClient(scored_chunks))
        result = retriever.search("test_dataset", "test_model", "query", top_k=2)

        assert len(result) == 2
        texts = {sc.indexed_chunk.text for sc in result}
        assert "high similarity" in texts
        assert "medium similarity" in texts
        assert "low similarity" not in texts
