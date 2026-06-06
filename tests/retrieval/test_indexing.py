from ritam.retrieval.indexing import Indexer, _create_chunks
from ritam.retrieval.types import ChunkingConfig, IndexerInput
from tests.fakes import FakeLlmClient


class TestChunking:
    def test_create_chunks_happy_path(self) -> None:
        text = "This is a sample document. It has several sentences. We will chunk it."
        chunk_size = 70
        chunk_separator = ". "
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size, chunk_separator=chunk_separator
        )
        file_path = "assets/docs/kubernetes_intro.md"

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
        file_path = "assets/docs/kubernetes_intro.md"

        chunks = _create_chunks(text, file_path, chunking_config)

        assert len(chunks) == 2
        assert chunks[0].text == "This is a sample document. It has several sentences."
        assert chunks[1].text == "We will chunk it."

    def test_create_chunks_empty_content_returns_empty_list(self) -> None:
        text = ""
        config = ChunkingConfig(chunk_size=10, chunk_separator="\\n\\n")
        chunks = _create_chunks(text, "assets/docs/whatever.md", config)
        assert chunks == []


class TestIndexer:
    def test_indexer_happy_path(self, fake_llm_client: FakeLlmClient) -> None:
        indexer = Indexer(
            embedding_model="models/embedding-001",
            chunking_config=ChunkingConfig(chunk_size=50, chunk_separator=". "),
        )

        indexer_input = [
            IndexerInput(
                doc_path="source/test.md",
                doc_content="This is a test document. It will be indexed.",
            )
        ]

        indexed_chunks = indexer.build_index(fake_llm_client, indexer_input)

        assert len(indexed_chunks) == 1
        assert indexed_chunks[0].source == "source/test.md#chunk-0"
        assert indexed_chunks[0].doc_path == "source/test.md"
        assert indexed_chunks[0].chunk_id == 0

    def test_indexer_source_is_path_not_object_repr(
        self, fake_llm_client: FakeLlmClient
    ) -> None:
        """Regression: source must be '<doc_path>#chunk-N', never the IndexerInput repr."""
        indexer = Indexer(
            embedding_model="models/embedding-001",
            chunking_config=ChunkingConfig(chunk_size=20, chunk_separator=". "),
        )

        indexer_input = [
            IndexerInput(
                doc_path="docs/a.md",
                doc_content="First sentence. Second sentence. Third one here.",
            )
        ]

        indexed_chunks = indexer.build_index(fake_llm_client, indexer_input)

        assert len(indexed_chunks) > 1
        for chunk_id, chunk in enumerate(indexed_chunks):
            assert chunk.source == f"docs/a.md#chunk-{chunk_id}"
            assert "doc_content" not in chunk.source
