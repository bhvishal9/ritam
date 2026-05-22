import json
import logging
import uuid
from collections.abc import Generator

import pytest
from _pytest.logging import LogCaptureFixture
from fastapi.testclient import TestClient

from llm_lab.api.dependencies import get_llm_client, get_retriever_client
from llm_lab.llm.errors import LlmUnavailableError
from llm_lab.main import app
from llm_lab.vector_store.types import IndexedChunk, ScoredChunk
from tests.fakes import FakeLlmClient, FakeRetriever


def _make_fake_chunks() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            score=0.95,
            indexed_chunk=IndexedChunk(
                text="Chunk about Kubernetes pods",
                source="assets/docs/kubernetes_intro.md",
                embedding=[1.0, 0.0],
                chunk_id=0,
                doc_path="assets/docs/kubernetes_intro.md",
            ),
        ),
        ScoredChunk(
            score=0.80,
            indexed_chunk=IndexedChunk(
                text="Some other chunk",
                source="assets/docs/other.md",
                embedding=[0.0, 1.0],
                chunk_id=1,
                doc_path="assets/docs/other.md",
            ),
        ),
    ]


class TestQueryApi:
    @pytest.fixture(autouse=True)
    def clear_dependency_overrides(self) -> Generator[None]:
        yield
        app.dependency_overrides.clear()

    def test_query_happy_path(self, client: TestClient) -> None:
        fake_chunks = _make_fake_chunks()
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever(
            chunks=fake_chunks
        )
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient(
            response="fake answer from LLM"
        )

        response = client.post(
            "/query",
            json={
                "dataset": "test_dataset",
                "query": "What is a Kubernetes pod?",
                "top_k": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "fake answer from LLM"
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) == 1
        first_source = data["sources"][0]
        assert first_source["source"] == "assets/docs/kubernetes_intro.md"
        assert first_source["chunk_id"] == 0

    def test_query_invalid_query_returns_400(self, client: TestClient) -> None:
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever()
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()

        response = client.post(
            "/query", json={"query": "", "top_k": 1, "dataset": "test_dataset"}
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Question must be a non-empty string"}

    def test_query_invalid_top_k_returns_400(self, client: TestClient) -> None:
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever()
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()

        response = client.post(
            "/query",
            json={"query": "Test Query", "top_k": 0, "dataset": "test_dataset"},
        )
        assert response.status_code == 400
        assert response.json() == {"error": "top_k must be between 1 and 10"}

    def test_query_missing_index_returns_500(self, client: TestClient) -> None:
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever(
            error=ValueError(
                "Dataset test_dataset not found, make sure to run the index command first"
            )
        )
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()

        response = client.post(
            "/query",
            json={"query": "Test Query", "top_k": 1, "dataset": "test_dataset"},
        )

        assert response.status_code == 500
        assert response.json() == {
            "error": "Dataset test_dataset not found, make sure to run the index command first"
        }

    def test_query_llm_unavailable_returns_502(self, client: TestClient) -> None:
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever(
            chunks=_make_fake_chunks()
        )
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient(
            generate_error=LlmUnavailableError("Fake client unavailable")
        )

        response = client.post(
            "/query",
            json={"query": "Test Query", "top_k": 1, "dataset": "test_dataset"},
        )

        assert response.status_code == 502
        assert response.json() == {"error": "Fake client unavailable"}

    def test_query_log_fields_exists(
        self, client: TestClient, caplog: LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="llm_lab.api")
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever(
            chunks=_make_fake_chunks()[:1]
        )
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient(
            response="fake answer from LLM"
        )

        client.post(
            "/query",
            json={
                "query": "What is a Kubernetes pod?",
                "top_k": 1,
                "dataset": "test_dataset",
            },
        )

        api_records = [r for r in caplog.records if r.name == "llm_lab.api"]
        assert len(api_records) == 1, (
            f"Expected 1 API log record, got: {caplog.messages}"
        )
        logs = json.loads(api_records[0].message)
        assert logs["request_id"] != ""
        assert logs["request_id"] != "uuid-not-set"
        uuid.UUID(logs["request_id"])
        assert logs["top_k"] == 1
        assert logs["dataset"] == "test_dataset"
