import uuid
from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient, models

from ritam.vector_store.errors import IndexNotFoundError, IndexSchemaMismatchError
from ritam.vector_store.qdrant import (
    QdrantStoreClient,
    _build_collection_name,
    _create_collection,
)
from ritam.vector_store.types import IndexedChunk

# A threshold low enough that the fake reranker never filters anything out, so
# tests that are about retrieval aren't accidentally testing the threshold.
KEEP_ALL = -1_000.0

# Helpers
# ---------------------------------------------------------------------------


class FakeCrossEncoder:
    """Deterministic reranker: 'chunk text N' scores -N, so lower N ranks higher.

    Keeps the tests free of a real model download and makes the expected
    ordering explicit rather than dependent on a neural network.
    """

    def rerank(
        self, query: str, documents: list[str], **kwargs: Any
    ) -> Iterable[float]:
        return [-float(document.rsplit(" ", 1)[-1]) for document in documents]


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
        client = QdrantStoreClient(
            client_url="http://unused",
            api_key="api_key",
            text_encoder=FakeCrossEncoder(),
        )
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
    def test_creates_named_dense_and_sparse_vectors(
        self, in_memory_qdrant: QdrantClient
    ) -> None:
        _create_collection(in_memory_qdrant, "test-collection", embedding_size=4)

        params = in_memory_qdrant.get_collection("test-collection").config.params
        assert isinstance(params.vectors, dict)
        assert params.vectors["dense"].size == 4
        assert params.sparse_vectors is not None
        assert "sparse" in params.sparse_vectors

    def test_is_idempotent(self, in_memory_qdrant: QdrantClient) -> None:
        _create_collection(in_memory_qdrant, "test-collection", embedding_size=4)
        # Calling again must not raise
        _create_collection(in_memory_qdrant, "test-collection", embedding_size=4)

        assert in_memory_qdrant.collection_exists("test-collection")

    def test_rejects_pre_hybrid_collection(
        self, in_memory_qdrant: QdrantClient
    ) -> None:
        # A collection built the old way: one unnamed vector, no sparse config.
        in_memory_qdrant.create_collection(
            "legacy-collection",
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )

        with pytest.raises(IndexSchemaMismatchError, match="predates hybrid search"):
            _create_collection(in_memory_qdrant, "legacy-collection", embedding_size=4)


# ---------------------------------------------------------------------------
# QdrantStoreClient.store
# ---------------------------------------------------------------------------


class TestQdrantStoreClientStore:
    def test_store_creates_collection_automatically(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunks = [_make_chunk(0, [1.0, 0.0])]
        store_client.store(chunks, dataset="ds", embedding_model="test-model")

        assert store_client.client.collection_exists("test-model")

    def test_store_sanitizes_embedding_model_for_collection_name(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunks = [_make_chunk(0, [1.0, 0.0])]
        store_client.store(
            chunks,
            dataset="ds",
            embedding_model="gemini-embedding/001",
        )

        assert store_client.client.collection_exists("gemini-embedding-001")
        assert not store_client.client.collection_exists("gemini-embedding/001")

    def test_store_persists_all_payload_fields(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunk = _make_chunk(
            0, [1.0, 0.0], source="docs/a.md#chunk-0", doc_path="docs/a.md"
        )
        store_client.store([chunk], dataset="my_ds", embedding_model="test-model")

        results = store_client.client.scroll(
            "test-model", with_payload=True, with_vectors=True
        )
        points = results[0]
        assert len(points) == 1
        payload = points[0].payload
        assert payload is not None
        assert payload["dataset"] == "my_ds"
        assert payload["text"] == "chunk text 0"
        assert payload["source"] == "docs/a.md#chunk-0"
        assert payload["chunk_id"] == 0
        assert payload["doc_path"] == "docs/a.md"

    def test_store_uses_deterministic_point_id(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunk = _make_chunk(0, [1.0, 0.0], source="docs/a.md#chunk-0")
        store_client.store([chunk], dataset="ds", embedding_model="test-model")

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
        store_client.store([chunk], dataset="ds", embedding_model="test-model")
        store_client.store([chunk], dataset="ds", embedding_model="test-model")

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
                query="find the chunk",
                query_embedding=[1.0, 0.0],
                limit=5,
                top_k=5,
                reranking_threshold=KEEP_ALL,
            )

    def test_query_raises_when_dataset_not_indexed(
        self, store_client: QdrantStoreClient
    ) -> None:
        # Collection exists (dataset_a was indexed) but dataset_b never was, so
        # the empty result is "not indexed", not "nothing relevant".
        store_client.store(
            [_make_chunk(0, [1.0, 0.0])],
            dataset="dataset_a",
            embedding_model="test-model",
        )

        with pytest.raises(IndexNotFoundError, match="has not been indexed"):
            store_client.query(
                dataset="dataset_b",
                embedding_model="test-model",
                query="find the chunk",
                query_embedding=[1.0, 0.0],
                limit=5,
                top_k=5,
                reranking_threshold=KEEP_ALL,
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
        store_client.store(chunks, dataset="ds", embedding_model="test-model")

        results = store_client.query(
            dataset="ds",
            embedding_model="test-model",
            query="find the chunk",
            query_embedding=[1.0, 0.0],
            limit=2,
            top_k=2,
            reranking_threshold=KEEP_ALL,
        )

        assert len(results) == 2
        assert results[0].score > results[1].score
        assert results[0].indexed_chunk.chunk_id == 0

    def test_query_respects_limit(self, store_client: QdrantStoreClient) -> None:
        chunks = [_make_chunk(i, [1.0, 0.0]) for i in range(5)]
        store_client.store(chunks, dataset="ds", embedding_model="test-model")

        results = store_client.query(
            dataset="ds",
            embedding_model="test-model",
            query="find the chunk",
            query_embedding=[1.0, 0.0],
            limit=2,
            top_k=2,
            reranking_threshold=KEEP_ALL,
        )

        assert len(results) == 2

    def test_query_filters_by_dataset(self, store_client: QdrantStoreClient) -> None:
        chunk_a = _make_chunk(0, [1.0, 0.0], source="a.md#0")
        chunk_b = _make_chunk(1, [1.0, 0.0], source="b.md#0")
        store_client.store([chunk_a], dataset="dataset_a", embedding_model="test-model")
        store_client.store([chunk_b], dataset="dataset_b", embedding_model="test-model")

        results = store_client.query(
            dataset="dataset_a",
            embedding_model="test-model",
            query="find the chunk",
            query_embedding=[1.0, 0.0],
            limit=10,
            top_k=10,
            reranking_threshold=KEEP_ALL,
        )

        assert len(results) == 1
        assert results[0].indexed_chunk.source == "a.md#0"

    def test_query_maps_payload_to_indexed_chunk(
        self, store_client: QdrantStoreClient
    ) -> None:
        chunk = _make_chunk(
            7, [1.0, 0.0], source="docs/k8s.md#chunk-7", doc_path="docs/k8s.md"
        )
        store_client.store([chunk], dataset="ds", embedding_model="test-model")

        results = store_client.query(
            dataset="ds",
            embedding_model="test-model",
            query="find the chunk",
            query_embedding=[1.0, 0.0],
            limit=1,
            top_k=1,
            reranking_threshold=KEEP_ALL,
        )

        assert len(results) == 1
        ic = results[0].indexed_chunk
        assert ic.text == "chunk text 7"
        assert ic.source == "docs/k8s.md#chunk-7"
        assert ic.chunk_id == 7
        assert ic.doc_path == "docs/k8s.md"
