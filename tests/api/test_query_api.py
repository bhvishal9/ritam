import json
import logging
import uuid
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from ritam.api.dependencies import get_llm_client, get_retriever_client
from ritam.llm.errors import LlmUnavailableError
from ritam.main import app
from ritam.observability.logging import JsonFormatter
from ritam.vector_store.errors import IndexNotFoundError
from ritam.vector_store.types import IndexedChunk, ScoredChunk
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
                "embedding_model": "test-embed",
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
            "/query",
            json={
                "query": "",
                "top_k": 1,
                "dataset": "test_dataset",
                "embedding_model": "test-embed",
            },
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Question must be a non-empty string"}

    def test_query_invalid_top_k_returns_400(self, client: TestClient) -> None:
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever()
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()

        response = client.post(
            "/query",
            json={
                "query": "Test Query",
                "top_k": 0,
                "dataset": "test_dataset",
                "embedding_model": "test-embed",
            },
        )
        assert response.status_code == 400
        assert response.json() == {"error": "top_k must be between 1 and 10"}

    def test_query_missing_index_returns_404(self, client: TestClient) -> None:
        app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever(
            error=IndexNotFoundError(
                "No index found for embedding model 'test-embed'; "
                "run the index command for this dataset first."
            )
        )
        app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()

        response = client.post(
            "/query",
            json={
                "query": "Test Query",
                "top_k": 1,
                "dataset": "test_dataset",
                "embedding_model": "test-embed",
            },
        )

        assert response.status_code == 404
        assert response.json() == {
            "error": "No index found for embedding model 'test-embed'; "
            "run the index command for this dataset first."
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
            json={
                "query": "Test Query",
                "top_k": 1,
                "dataset": "test_dataset",
                "embedding_model": "test-embed",
            },
        )

        assert response.status_code == 502
        assert response.json() == {"error": "Fake client unavailable"}

    def test_middleware_logs_when_handler_raises(self) -> None:
        # Use raise_server_exceptions=False so Starlette converts the unhandled
        # exception into a 500 response (as it would in production) rather than
        # re-raising into the test. The middleware's finally block must still
        # emit request_complete with status_code=500.
        emitted: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.name == "ritam.api.middleware":
                    emitted.append(self.format(record))

        handler = CaptureHandler()
        handler.setFormatter(JsonFormatter())
        logging.getLogger().addHandler(handler)
        try:
            app.dependency_overrides[get_retriever_client] = lambda: FakeRetriever(
                error=RuntimeError("boom")
            )
            app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()
            with TestClient(app, raise_server_exceptions=False) as local_client:
                response = local_client.post(
                    "/query",
                    json={
                        "query": "What is a Kubernetes pod?",
                        "top_k": 1,
                        "dataset": "test_dataset",
                        "embedding_model": "test-embed",
                    },
                )
        finally:
            logging.getLogger().removeHandler(handler)

        assert response.status_code == 500
        assert len(emitted) == 1, f"Expected 1 API log record, got: {emitted}"
        logs = json.loads(emitted[0])
        assert logs["message"] == "request_complete"
        assert logs["status_code"] == 500
        # error_type is set only when the exception propagates past Starlette's
        # error middleware into ours. Starlette converts to a 500 response
        # first, so we don't see the raise here — but we DO see status_code=500
        # via wrapped_send, which is the primary guarantee that matters.

    def test_query_log_fields_exists(self, client: TestClient) -> None:
        # Capture formatted JSON output at emit time. Reformatting later won't
        # work because contextvars (request_id, top_k, dataset) are reset once
        # the request task ends.
        emitted: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.name == "ritam.api.middleware":
                    emitted.append(self.format(record))

        handler = CaptureHandler()
        handler.setFormatter(JsonFormatter())
        logging.getLogger().addHandler(handler)
        try:
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
                    "embedding_model": "test-embed",
                },
            )
        finally:
            logging.getLogger().removeHandler(handler)

        assert len(emitted) == 1, f"Expected 1 API log record, got: {emitted}"
        logs = json.loads(emitted[0])
        assert logs["message"] == "request_complete"
        assert logs["request_id"] != ""
        assert logs["request_id"] != "not-set"
        uuid.UUID(logs["request_id"])
        assert logs["top_k"] == 1
        assert logs["dataset"] == "test_dataset"
        assert logs["status_code"] == 200
