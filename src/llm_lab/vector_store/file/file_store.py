import math
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from llm_lab.config.paths import DEFAULT_DESTINATION_DIR
from llm_lab.vector_store.file.types import IndexFile, ManifestFile, ManifestIndexFile
from llm_lab.vector_store.types import IndexedChunk, ScoredChunk, VectorStoreClient

MAX_CHUNKS_PER_INDEX_FILE = 10


def _create_dest_dir(dest_dir: Path) -> None:
    """Create destination directory, removing it first if it exists."""
    try:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, PermissionError) as e:
        raise ValueError(
            f"Could not create destination directory {dest_dir}: {e}"
        ) from e


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same length")

    dot_product = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        # should not happen with real embeddings, but guard anyway
        return 0.0

    return dot_product / (norm_a * norm_b)


def _load_manifest(manifest_path: Path) -> ManifestFile:
    """Load and validate a manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found at {manifest_path}, make sure to index the dataset first."
        )
    manifest_data = manifest_path.read_text(encoding="utf-8")
    if not manifest_data:
        raise ValueError(
            f"Manifest file at {manifest_path} is empty, make sure to index the dataset first."
        )
    try:
        return ManifestFile.model_validate_json(manifest_data)
    except ValidationError as err:
        raise ValueError(
            f"Manifest file at {manifest_path} is malformed: {err}"
        ) from err


def _load_indexed_chunks(
    indexed_chunks_dir: Path, manifest_path: Path
) -> list[IndexedChunk]:
    """Load indexed chunks from the dataset directory."""
    manifest = _load_manifest(manifest_path)
    indexed_chunks = []
    for index_file in manifest.index_files:
        index_file_path = indexed_chunks_dir / index_file.path
        if not index_file_path.exists():
            raise FileNotFoundError(
                f"Index file {index_file_path} not found, make sure to index the dataset first."
            )
        try:
            index_file_data = index_file_path.read_text(encoding="utf-8")
            index_file_validated_data = IndexFile.model_validate_json(index_file_data)
        except ValidationError as err:
            raise ValueError(
                f"Index file at {index_file_path} is malformed: {err}"
            ) from err
        index_file_chunks = index_file_validated_data.chunks
        indexed_chunks.extend(index_file_chunks)
    return indexed_chunks


class FileStoreClient(VectorStoreClient):
    """File-based implementation of VectorStoreClient."""

    def __init__(self, dest_dir: Path = DEFAULT_DESTINATION_DIR) -> None:
        self.dest_dir = dest_dir

    def store(
        self,
        indexed_chunks: list[IndexedChunk],
        dataset: str,
        embedding_model: str,
        docs_count: int,
    ) -> None:
        """Store the indexed chunks into a file based indexed chunk store."""
        manifest_file = self.dest_dir / dataset / "manifest.json"
        index_creation_dir = self.dest_dir / dataset / "indexes"
        _create_dest_dir(index_creation_dir)
        timestamp = datetime.now(tz=UTC)
        file_counter = 0
        manifest_index_files = []
        for idx in range(0, len(indexed_chunks), MAX_CHUNKS_PER_INDEX_FILE):
            chunk_slice = indexed_chunks[idx : idx + MAX_CHUNKS_PER_INDEX_FILE]
            index_id = f"index-{file_counter:04}"
            index_file_name = f"{index_id}.json"
            index_path = index_creation_dir / index_file_name
            index_data = IndexFile(
                index_id=index_id,
                chunks=chunk_slice,
            )
            index_path.write_text(index_data.model_dump_json(indent=2))
            manifest_index_files.append(
                ManifestIndexFile(
                    index_id=index_id,
                    path=str(index_path.relative_to(index_creation_dir)),
                    num_chunks=len(chunk_slice),
                )
            )
            file_counter += 1  # noqa: SIM113
        manifest = ManifestFile(
            dataset=dataset,
            embedding_model=embedding_model,
            created_at=timestamp,
            total_docs=docs_count,
            total_chunks=len(indexed_chunks),
            index_files=manifest_index_files,
        )
        manifest_file.write_text(manifest.model_dump_json(indent=2))

    def query(
        self,
        dataset: str,
        embedding_model: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[ScoredChunk]:
        """Query the vector store and return a list of the top_k most relevant chunks."""
        manifest_file = self.dest_dir / dataset / "manifest.json"
        index_creation_dir = self.dest_dir / dataset / "indexes"
        scored_chunks = []
        indexed_chunks = _load_indexed_chunks(index_creation_dir, manifest_file)
        for chunk in indexed_chunks:
            score = _cosine_similarity(query_embedding, chunk.embedding)
            scored_chunks.append(
                ScoredChunk(
                    score=score,
                    indexed_chunk=chunk,
                )
            )
        sorted_chunks = sorted(scored_chunks, key=lambda x: x.score, reverse=True)[
            :limit
        ]
        return sorted_chunks

    def delete(self, dataset: str, embedding_model: str, doc_path: str) -> None:
        """Not implemented — FileStoreClient is deprecated in favour of Qdrant."""
        raise NotImplementedError("FileStoreClient does not support delete")
