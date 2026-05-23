from pathlib import Path

from llm_lab.config.paths import BASE_DIR
from llm_lab.llm.types import LlmClient
from llm_lab.retrieval.types import (
    ChunkingConfig,
)
from llm_lab.vector_store.types import Chunk, IndexedChunk


def _create_chunks(
    file_content: str, file_name: Path, chunking_config: ChunkingConfig
) -> list[Chunk]:
    """Create chunks from a file content."""
    chunks = []
    separator = chunking_config.chunk_separator
    chunk_size = chunking_config.chunk_size
    content_length = len(file_content)
    start = 0
    while start < content_length:
        end = min(start + chunk_size, content_length)
        if end < content_length:
            sep_index = file_content.rfind(separator, start, end)
            if sep_index != -1 and sep_index > start:
                end = sep_index + len(separator)

        chunk_text = file_content[start:end].strip()
        if chunk_text:
            chunk = Chunk(
                text=chunk_text,
                doc_path=str(file_name),
            )
            chunks.append(chunk)
        start = end

    return chunks


def _read_file(file_path: Path) -> str:
    """Read a file and return its content."""
    try:
        file_content = file_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as err:
        raise ValueError(f"File {file_path} not found: {err}") from err
    if not file_content:
        raise ValueError(f"File {file_path} is empty")
    return file_content


class Indexer:
    """Indexer to create embeddings for documents in a directory."""

    def __init__(
        self,
        embedding_model: str,
        chunking_config: ChunkingConfig,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunking_config = chunking_config

    def build_index(self, llm_client: LlmClient, docs: list[str]) -> list[IndexedChunk]:
        """Build index by creating embeddings for document chunks."""
        indexed_chunks = []
        for doc in docs:
            file_content = _read_file(BASE_DIR / doc)
            chunks = _create_chunks(file_content, BASE_DIR / doc, self.chunking_config)
            for chunk_id, chunk in enumerate(chunks):
                embedding = llm_client.embed_text(chunk.text, self.embedding_model)
                source = f"{doc}#chunk-{chunk_id}"
                indexed_chunk = IndexedChunk(
                    text=chunk.text,
                    doc_path=chunk.doc_path,
                    source=source,
                    embedding=embedding,
                    chunk_id=chunk_id,
                )
                indexed_chunks.append(indexed_chunk)
        return indexed_chunks
