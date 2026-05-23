from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

import llm_lab.retrieval.indexing as indexing
from llm_lab.retrieval.indexing import Indexer, _create_chunks
from llm_lab.retrieval.types import ChunkingConfig
from tests.fakes import FakeLlmClient


class TestChunking:
    def test_create_chunks_happy_path(self) -> None:
        text = "This is a sample document. It has several sentences. We will chunk it."
        chunk_size = 70
        chunk_separator = ". "
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size, chunk_separator=chunk_separator
        )
        file_path = Path("assets/docs/kubernetes_intro.md")

        chunks = _create_chunks(text, file_path, chunking_config)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].doc_path == "assets/docs/kubernetes_intro.md"

    def test_create_chunks_splits_on_separator_before_limit(self) -> None:
        text = "This is a sample document. It has several sentences. We will chunk it."
        chunk_size = 65
        chunk_separator = ". "
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size, chunk_separator=chunk_separator
        )
        file_path = Path("assets/docs/kubernetes_intro.md")

        chunks = _create_chunks(text, file_path, chunking_config)

        assert len(chunks) == 2
        assert chunks[0].text == "This is a sample document. It has several sentences."
        assert chunks[1].text == "We will chunk it."

    def test_create_chunks_empty_content_returns_empty_list(self) -> None:
        text = ""
        config = ChunkingConfig(chunk_size=10, chunk_separator="\\n\\n")
        chunks = _create_chunks(text, Path("assets/docs/whatever.md"), config)
        assert chunks == []


class TestIndexer:
    def test_indexer_happy_path(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, fake_llm_client: FakeLlmClient
    ) -> None:
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        file_path = source_dir / "test.md"
        file_path.write_text(
            "This is a test document. It will be indexed.", encoding="utf-8"
        )

        monkeypatch.setattr(indexing, "BASE_DIR", tmp_path)

        indexer = Indexer(
            embedding_model="models/embedding-001",
            chunking_config=ChunkingConfig(chunk_size=50, chunk_separator=". "),
        )

        indexed_chunks = indexer.build_index(fake_llm_client, ["source/test.md"])
        assert len(indexed_chunks) == 1

    def test_indexer_missing_file_raises_error(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        fake_llm_client: FakeLlmClient,
    ) -> None:
        monkeypatch.setattr(indexing, "BASE_DIR", tmp_path)

        indexer = Indexer(
            embedding_model="models/embedding-001",
            chunking_config=ChunkingConfig(chunk_size=50, chunk_separator=". "),
        )

        with pytest.raises(ValueError, match="not found"):
            indexer.build_index(fake_llm_client, ["nonexistent/doc.md"])
