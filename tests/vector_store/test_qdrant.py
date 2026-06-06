import uuid
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient

from ritam.vector_store.errors import IndexNotFoundError
from ritam.vector_store.qdrant import (
    QdrantStoreClient,
    _build_collection_name,
    _create_collection,
)
from ritam.vector_store.types import IndexedChunk

# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: int,
    embedding: list[float],
    source: str | None = None,
    doc_path: str = "docs/test.md",
) -> IndexedChunk:
    return IndexedChunk(
        text=f"chunk text {chunk_id}",
        doc_path=doc_path,
        source=source or f"docs/test.md#chunk-{chunk_id}",
        chunk_id=chunk_id,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_qdrant() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def store_client(in_memory_qdrant: QdrantClient) -> QdrantStoreClient:
    """QdrantStoreClient wired to the in-memory QdrantClient."""
    with patch("ritam.vector_store.qdrant.QdrantClient", return_value=in_memory_qdrant):
        client = QdrantStoreClient(client_url="http://unused", api_key="api_key")
    return client


# ---------------------------------------------------------------------------
# _build_collection_name
# ---------------------------------------------------------------------------


class TestBuildCollectionName:
    def test_alphanumeric_unchanged(self) -> None:
        assert _build_collection_name("abc123") == "abc123"

    def test_uppercase_lowercased(self) -> None:
        assert _build_collection_name("GeminiEmbedding") == "geminiembedding"

    def test_special_chars_replaced_with_dash(self) -> None:
        assert _build_collection_name("gemini-embedding-001") == "gemini-embedding-001"

    def test_dots_and_slashes_replaced(self) -> None:
        assert _build_collection_name("text-embedding/v3.0") == "text-embedding-v3-0"


# ---------------------------------------------------------------------------
# _create_collection
# ---------------------------------------------------------------------------


class TestCreateCollection:
    def test_creates_collection_with_correct_vector_size(
        self, in_memory_qdrant: QdrantClient
    ) -> None:
        _create_collection(in_memory_qdrant, "test-collection", embedding_size=4)

        info = in_memory_qdrant.get_collection("test-collection")
        assert info.config.params.vectors.size == 4

    def test_is_idempotent(self, in_memory_qdrant: QdrantClient) -> None:
        _create_collection(in_memory_qdrant, "test-collection", embedding_size=4)
        # Calling again must not raise
        _create_collection(in_memory_qdrant, "test-collection", embedding_size=4)

        assert in_memory_qdrant.collection_exists("test-collection")


# ---------------------------------------------------------------------------
# QdrantStoreClient.store
# ---------------------------------------------------------------------------


class TestQdrantStoreClientStore:
    def test_store_creates_collection_automatically(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunks = [_make_chunk(0, [1.0, 0.0])]
        store_client.store(
            chunks, dataset="ds", embedding_model="test-model", docs_count=1
        )

        assert store_client.client.collection_exists("test-model")

    def test_store_sanitizes_embedding_model_for_collection_name(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunks = [_make_chunk(0, [1.0, 0.0])]
        store_client.store(
            chunks,
            dataset="ds",
            embedding_model="gemini-embedding/001",
            docs_count=1,
        )

        assert store_client.client.collection_exists("gemini-embedding-001")
        assert not store_client.client.collection_exists("gemini-embedding/001")

    def test_store_persists_all_payload_fields(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunk = _make_chunk(
            0, [1.0, 0.0], source="docs/a.md#chunk-0", doc_path="docs/a.md"
        )
        store_client.store(
            [chunk], dataset="my_ds", embedding_model="test-model", docs_count=1
        )

        results = store_client.client.scroll(
            "test-model", with_payload=True, with_vectors=True
        )
        points = results[0]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["dataset"] == "my_ds"
        assert payload["text"] == "chunk text 0"
        assert payload["source"] == "docs/a.md#chunk-0"
        assert payload["chunk_id"] == 0
        assert payload["doc_path"] == "docs/a.md"

    def test_store_uses_deterministic_point_id(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunk = _make_chunk(0, [1.0, 0.0], source="docs/a.md#chunk-0")
        store_client.store(
            [chunk], dataset="ds", embedding_model="test-model", docs_count=1
        )

        expected_id = uuid.uuid5(
            namespace=uuid.NAMESPACE_DNS, name="ds-test-model-docs/a.md#chunk-0"
        )
        results = store_client.client.scroll("test-model", with_payload=True)
        point_id = results[0][0].id
        assert point_id == str(expected_id)

    def test_store_upsert_does_not_duplicate(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunk = _make_chunk(0, [1.0, 0.0])
        store_client.store(
            [chunk], dataset="ds", embedding_model="test-model", docs_count=1
        )
        store_client.store(
            [chunk], dataset="ds", embedding_model="test-model", docs_count=1
        )

        count = store_client.client.count("test-model").count
        assert count == 1


# ---------------------------------------------------------------------------
# QdrantStoreClient.query
# ---------------------------------------------------------------------------


class TestQdrantStoreClientQuery:
    def test_query_raises_when_collection_missing(
        self, store_client: QdrantStoreClient
    ) -> None:
        with pytest.raises(IndexNotFoundError, match="No index found"):
            store_client.query(
                dataset="ds",
                embedding_model="nonexistent-model",
                query_embedding=[1.0, 0.0],
                limit=5,
            )

    def test_query_returns_results_sorted_by_score(
        self, store_client: QdrantStoreClient
    ) -> None:
        # chunk 0: parallel to query → highest cosine score
        # chunk 1: orthogonal to query → score 0
        chunks = [
            _make_chunk(0, [1.0, 0.0]),
            _make_chunk(1, [0.0, 1.0]),
        ]
        store_client.store(
            chunks, dataset="ds", embedding_model="test-model", docs_count=1
        )

        results = store_client.query(
            dataset="ds",
            embedding_model="test-model",
            query_embedding=[1.0, 0.0],
            limit=2,
        )

        assert len(results) == 2
        assert results[0].score > results[1].score
        assert results[0].indexed_chunk.chunk_id == 0

    def test_query_respects_limit(self, store_client: QdrantStoreClient) -> None:
        chunks = [_make_chunk(i, [1.0, 0.0]) for i in range(5)]
        store_client.store(
            chunks, dataset="ds", embedding_model="test-model", docs_count=1
        )

        results = store_client.query(
            dataset="ds",
            embedding_model="test-model",
            query_embedding=[1.0, 0.0],
            limit=2,
        )

        assert len(results) == 2

    def test_query_filters_by_dataset(self, store_client: QdrantStoreClient) -> None:
        chunk_a = _make_chunk(0, [1.0, 0.0], source="a.md#0")
        chunk_b = _make_chunk(1, [1.0, 0.0], source="b.md#0")
        store_client.store(
            [chunk_a], dataset="dataset_a", embedding_model="test-model", docs_count=1
        )
        store_client.store(
            [chunk_b], dataset="dataset_b", embedding_model="test-model", docs_count=1
        )

        results = store_client.query(
            dataset="dataset_a",
            embedding_model="test-model",
            query_embedding=[1.0, 0.0],
            limit=10,
        )

        assert len(results) == 1
        assert results[0].indexed_chunk.source == "a.md#0"

    def test_query_maps_payload_to_indexed_chunk(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunk = _make_chunk(
            7, [1.0, 0.0], source="docs/k8s.md#chunk-7", doc_path="docs/k8s.md"
        )
        store_client.store(
            [chunk], dataset="ds", embedding_model="test-model", docs_count=1
        )

        results = store_client.query(
            dataset="ds",
            embedding_model="test-model",
            query_embedding=[1.0, 0.0],
            limit=1,
        )

        assert len(results) == 1
        ic = results[0].indexed_chunk
        assert ic.text == "chunk text 7"
        assert ic.source == "docs/k8s.md#chunk-7"
        assert ic.chunk_id == 7
        assert ic.doc_path == "docs/k8s.md"
