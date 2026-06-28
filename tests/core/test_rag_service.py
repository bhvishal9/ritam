from ritam.core.rag_service import RagService
from ritam.retrieval.retriever import Retriever
from tests.fakes import FakeVectorStoreClient, NoCallLlmClient


class TestRagService:
    def test_rag_service_short_circuits_when_no_chunks(
        self, no_call_llm_client: NoCallLlmClient
    ) -> None:
        retriever = Retriever(
            no_call_llm_client, FakeVectorStoreClient(), similarity_threshold=0.7
        )
        rag_service = RagService(no_call_llm_client, retriever)

        result = rag_service.answer_question(
            dataset="test_dataset",
            embedding_model="test_model",
            query="nonsense query that should match nothing",
            top_k=3,
        )

        assert result.answer == "No relevant information found to answer the question."
        assert result.chunks == []
