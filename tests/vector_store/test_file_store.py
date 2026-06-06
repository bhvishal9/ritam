import json
from pathlib import Path

import pytest

from ritam.vector_store.file.file_store import (
    MAX_CHUNKS_PER_INDEX_FILE,
    FileStoreClient,
)
from ritam.vector_store.types import IndexedChunk


def _make_chunk(chunk_id: int, embedding: list[float]) -> IndexedChunk:
    return IndexedChunk(
        text=f"chunk text {chunk_id}",
        doc_path="docs/test.md",
        source=f"docs/test.md#chunk-{chunk_id}",
        chunk_id=chunk_id,
        embedding=embedding,
    )


class TestFileStoreClientStore:
    def test_store_creates_manifest_with_correct_metadata(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(i, [1.0, 0.0]) for i in range(3)]
        client = FileStoreClient(dest_dir=tmp_path)

        client.store(
            chunks,
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            docs_count=2,
        )

        manifest_path = tmp_path / "my_ds" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["dataset"] == "my_ds"
        assert manifest["embedding_model"] == "gemini-embedding-001"
        assert manifest["total_docs"] == 2
        assert manifest["total_chunks"] == 3

    def test_store_batches_chunks_into_multiple_index_files(
        self, tmp_path: Path
    ) -> None:
        chunk_count = MAX_CHUNKS_PER_INDEX_FILE + 3  # forces two files
        chunks = [_make_chunk(i, [1.0, 0.0]) for i in range(chunk_count)]
        client = FileStoreClient(dest_dir=tmp_path)

        client.store(
            chunks,
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            docs_count=1,
        )

        index_files = list((tmp_path / "my_ds" / "indexes").glob("*.json"))
        assert len(index_files) == 2

        manifest = json.loads((tmp_path / "my_ds" / "manifest.json").read_text())
        assert len(manifest["index_files"]) == 2
        assert manifest["index_files"][0]["num_chunks"] == MAX_CHUNKS_PER_INDEX_FILE
        assert manifest["index_files"][1]["num_chunks"] == 3

    def test_store_overwrites_existing_dataset(self, tmp_path: Path) -> None:
        client = FileStoreClient(dest_dir=tmp_path)
        client.store(
            [_make_chunk(0, [1.0, 0.0])],
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            docs_count=1,
        )

        client.store(
            [_make_chunk(0, [1.0, 0.0]), _make_chunk(1, [0.0, 1.0])],
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            docs_count=2,
        )

        manifest = json.loads((tmp_path / "my_ds" / "manifest.json").read_text())
        assert manifest["total_chunks"] == 2


class TestFileStoreClientQuery:
    def test_query_returns_chunks_sorted_by_similarity(self, tmp_path: Path) -> None:
        # chunk 0: parallel to query → highest score
        # chunk 1: orthogonal to query → score 0
        chunks = [
            _make_chunk(0, [1.0, 0.0]),
            _make_chunk(1, [0.0, 1.0]),
        ]
        client = FileStoreClient(dest_dir=tmp_path)
        client.store(
            chunks,
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            docs_count=1,
        )

        results = client.query(
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            query_embedding=[1.0, 0.0],
            limit=2,
        )

        assert len(results) == 2
        assert results[0].score > results[1].score
        assert results[0].indexed_chunk.chunk_id == 0

    def test_query_respects_limit(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(i, [1.0, 0.0]) for i in range(5)]
        client = FileStoreClient(dest_dir=tmp_path)
        client.store(
            chunks,
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            docs_count=1,
        )

        results = client.query(
            dataset="my_ds",
            embedding_model="gemini-embedding-001",
            query_embedding=[1.0, 0.0],
            limit=2,
        )

        assert len(results) == 2

    def test_query_raises_when_manifest_missing(self, tmp_path: Path) -> None:
        client = FileStoreClient(dest_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            client.query(
                dataset="nonexistent",
                embedding_model="gemini-embedding-001",
                query_embedding=[1.0, 0.0],
                limit=3,
            )

    def test_query_raises_when_manifest_malformed(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "my_ds"
        dataset_dir.mkdir()
        (dataset_dir / "manifest.json").write_text("not valid json{", encoding="utf-8")

        client = FileStoreClient(dest_dir=tmp_path)
        with pytest.raises(ValueError, match="malformed"):
            client.query(
                dataset="my_ds",
                embedding_model="gemini-embedding-001",
                query_embedding=[1.0, 0.0],
                limit=3,
            )
