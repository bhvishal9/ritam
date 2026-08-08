import re
import uuid

from qdrant_client import QdrantClient, models

from ritam.vector_store.errors import (
    IndexNotFoundError,
    IndexSchemaMismatchError,
    VectorStorePayloadError,
)
from ritam.vector_store.types import (
    IndexedChunk,
    ScoredChunk,
    TextReranker,
    VectorStoreClient,
)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def _build_collection_name(collection_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", collection_name).lower()


def _assert_hybrid_schema(client: QdrantClient, collection_name: str) -> None:
    """Fail loudly when an existing collection predates the hybrid schema.

    Collections built before named dense/sparse vectors carry a single unnamed
    vector. Upserting the new point shape into one of those fails deep inside
    the client with an unhelpful message, so check the shape here instead.
    """
    params = client.get_collection(collection_name).config.params
    vectors = params.vectors
    has_dense = isinstance(vectors, dict) and DENSE_VECTOR in vectors
    has_sparse = bool(params.sparse_vectors) and SPARSE_VECTOR in (
        params.sparse_vectors or {}
    )
    if has_dense and has_sparse:
        return
    raise IndexSchemaMismatchError(
        f"Collection '{collection_name}' predates hybrid search: it is missing "
        f"the '{DENSE_VECTOR}' and/or '{SPARSE_VECTOR}' vectors. Delete the "
        f"collection and re-index the datasets that used it."
    )


def _require_embedding(chunk: IndexedChunk) -> list[float]:
    """Indexing needs a dense vector; query results legitimately have none."""
    if chunk.embedding is None:
        raise VectorStorePayloadError(
            f"Chunk '{chunk.source}' has no embedding and cannot be indexed"
        )
    return chunk.embedding


def _create_collection(
    client: QdrantClient, collection_name: str, embedding_size: int
) -> None:
    if client.collection_exists(collection_name):
        _assert_hybrid_schema(client, collection_name)
        return
    try:
        client.create_collection(
            collection_name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True),
                    modifier=models.Modifier.IDF,
                )
            },
        )
        client.create_payload_index(
            collection_name,
            field_name="dataset",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name,
            field_name="doc_path",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception as err:
        raise RuntimeError(
            f"Failed to create collection {collection_name}: {err}"
        ) from err


class QdrantStoreClient(VectorStoreClient):
    def __init__(
        self, client_url: str, api_key: str | None, text_encoder: TextReranker
    ) -> None:
        self.client = QdrantClient(url=client_url, api_key=api_key)
        self.reranker = text_encoder

    def store(
        self,
        indexed_chunks: list[IndexedChunk],
        dataset: str,
        embedding_model: str,
    ) -> None:
        collection_name = _build_collection_name(embedding_model)
        embeddings = [_require_embedding(chunk) for chunk in indexed_chunks]
        # BM25 length normalisation compares each point against the average
        # length of the points in the collection — and a point here is a chunk.
        avg_chunk_length = sum(
            len(chunk.text.split()) for chunk in indexed_chunks
        ) / len(indexed_chunks)
        _create_collection(self.client, collection_name, len(embeddings[0]))
        points = []
        for chunk, embedding in zip(indexed_chunks, embeddings, strict=True):
            hash_id_text = f"{dataset}-{embedding_model}-{chunk.source}"
            point_id = uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=hash_id_text)
            points.append(
                models.PointStruct(
                    id=point_id,
                    payload={
                        "dataset": dataset,
                        "text": chunk.text,
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "doc_path": chunk.doc_path,
                    },
                    vector={
                        DENSE_VECTOR: embedding,
                        SPARSE_VECTOR: models.Document(
                            text=chunk.text,
                            model="Qdrant/bm25",
                            options={"avg_len": avg_chunk_length},
                        ),
                    },
                )
            )
        self.client.upsert(collection_name=collection_name, points=points)

    def delete(self, dataset: str, embedding_model: str, doc_path: str) -> None:
        collection_name = _build_collection_name(embedding_model)
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="dataset",
                            match=models.MatchValue(value=dataset),
                        ),
                        models.FieldCondition(
                            key="doc_path",
                            match=models.MatchValue(value=doc_path),
                        ),
                    ]
                )
            ),
        )

    def query(
        self,
        dataset: str,
        embedding_model: str,
        query: str,
        query_embedding: list[float],
        limit: int,
        top_k: int,
        reranking_threshold: float,
    ) -> list[ScoredChunk]:
        collection_name = _build_collection_name(embedding_model)
        if not self.client.collection_exists(collection_name):
            raise IndexNotFoundError(
                f"No index found for embedding model '{embedding_model}'; "
                "run the index command first."
            )
        dataset_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="dataset",
                    match=models.MatchValue(value=dataset),
                )
            ]
        )
        search_results = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using=SPARSE_VECTOR,
                    limit=limit,
                    filter=dataset_filter,
                ),
                models.Prefetch(
                    query=query_embedding,
                    using=DENSE_VECTOR,
                    limit=limit,
                    filter=dataset_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            # Nothing downstream reads the stored vectors back, and fetching
            # them would ship a full embedding per candidate on every query.
            with_vectors=False,
        ).points

        if not search_results:
            # The collection exists but this query returned nothing. Distinguish
            # "this dataset was never indexed" (caller error → surfaces as 404)
            # from "indexed, but nothing matched" (a valid empty answer). Pay for
            # this extra round-trip only on the empty branch, never on the hot
            # success path.
            dataset_count = self.client.count(
                collection_name=collection_name,
                count_filter=dataset_filter,
                exact=True,
            ).count
            if dataset_count == 0:
                raise IndexNotFoundError(
                    f"Dataset '{dataset}' has not been indexed for embedding "
                    f"model '{embedding_model}'; run the index command for it first."
                )

        payloads = []
        for point in search_results:
            if point.payload is None:
                raise VectorStorePayloadError(
                    f"Qdrant point {point.id} is missing its payload"
                )
            payloads.append(point.payload)
        candidate_texts = [payload["text"] for payload in payloads]
        reranked_scores = list(self.reranker.rerank(query, candidate_texts))
        scored_chunks = [
            ScoredChunk(
                score=score,
                indexed_chunk=IndexedChunk(
                    text=payload["text"],
                    source=payload["source"],
                    chunk_id=payload["chunk_id"],
                    doc_path=payload["doc_path"],
                ),
            )
            for payload, score in zip(payloads, reranked_scores, strict=True)
            if score >= reranking_threshold
        ]
        scored_chunks.sort(key=lambda sc: sc.score, reverse=True)
        return scored_chunks[:top_k]
