import re
import uuid

from qdrant_client import QdrantClient, models

from llm_lab.vector_store.errors import IndexNotFoundError, VectorStorePayloadError
from llm_lab.vector_store.types import IndexedChunk, ScoredChunk, VectorStoreClient


def _build_collection_name(collection_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", collection_name).lower()


def _create_collection(
    client: QdrantClient, collection_name: str, embedding_size: int
) -> None:
    if client.collection_exists(collection_name):
        return
    try:
        client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                # TO-DO: Get the dimension from the model
                size=embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name,
            field_name="dataset",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception as err:
        raise RuntimeError(
            f"Failed to create collection {collection_name}: {err}"
        ) from err


class QdrantStoreClient(VectorStoreClient):
    def __init__(self, client_url: str, api_key: str | None) -> None:
        self.client = QdrantClient(url=client_url, api_key=api_key)

    def store(
        self,
        indexed_chunks: list[IndexedChunk],
        dataset: str,
        embedding_model: str,
        docs_count: int,
    ) -> None:
        collection_name = _build_collection_name(embedding_model)
        embedding_size = len(indexed_chunks[0].embedding)
        _create_collection(self.client, collection_name, embedding_size)
        points = []
        for chunk in indexed_chunks:
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
                    vector=chunk.embedding,
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
        query_embedding: list[float],
        limit: int,
    ) -> list[ScoredChunk]:
        collection_name = _build_collection_name(embedding_model)
        if not self.client.collection_exists(collection_name):
            raise IndexNotFoundError(
                f"No index found for embedding model '{embedding_model}'; "
                "run the index command for this dataset first."
            )
        search_results = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="dataset",
                        match=models.MatchValue(value=dataset),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        ).points
        scored_chunks = []
        for point in search_results:
            payload = point.payload
            if payload is None:
                raise VectorStorePayloadError(
                    f"Qdrant point {point.id} is missing its payload"
                )
            scored_chunks.append(
                ScoredChunk(
                    score=point.score,
                    indexed_chunk=IndexedChunk(
                        text=payload["text"],
                        source=payload["source"],
                        chunk_id=payload["chunk_id"],
                        doc_path=payload["doc_path"],
                        embedding=point.vector,
                    ),
                )
            )
        return scored_chunks
