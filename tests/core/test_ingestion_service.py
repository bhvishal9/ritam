from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from sqlmodel import Session

import llm_lab.core.ingestion_service as ingestion
import llm_lab.retrieval.indexing as indexing
from llm_lab.core.ingestion_service import IngestionService
from llm_lab.retrieval.types import ChunkingConfig


class TestIngestionService:
    def test_ingest_docs(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Test ingestion service's ingest_docs method."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        file_path = source_dir / "test.md"
        file_path.write_text(
            "This is a test document. It will be indexed.", encoding="utf-8"
        )
        file_2_path = source_dir / "test2.md"
        file_2_path.write_text(
            "This is a test document. It will be indexed.", encoding="utf-8"
        )
        chunk_size = 70
        chunk_separator = ". "
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size, chunk_separator=chunk_separator
        )

        monkeypatch.setattr(indexing, "BASE_DIR", tmp_path)
        monkeypatch.setattr(ingestion, "BASE_DIR", tmp_path)

        ingestion_service = IngestionService(
            chunking_config=chunking_config,
            source_dir=source_dir,
            dataset="test",
            embedding_model="test",
            db_path=tmp_path / "test.db",
        )
        ingestion_diff = ingestion_service.ingest_docs()
        assert set(ingestion_diff.new_docs) == {"source/test2.md", "source/test.md"}
        assert ingestion_diff.deleted_docs == []
        assert ingestion_diff.updated_docs == []
        assert ingestion_diff.unchanged_docs == []

        with Session(ingestion_service.engine) as session:
            session.add(ingestion_diff.source_by_path["source/test.md"])
            session.add(ingestion_diff.source_by_path["source/test2.md"])
            session.commit()

        ingestion_diff = ingestion_service.ingest_docs()
        assert ingestion_diff.new_docs == []
        assert ingestion_diff.deleted_docs == []
        assert ingestion_diff.updated_docs == []
        assert set(ingestion_diff.unchanged_docs) == {
            "source/test2.md",
            "source/test.md",
        }

        file_path.write_text(
            "This is a test document. It will be indexed again.", encoding="utf-8"
        )
        ingestion_diff = ingestion_service.ingest_docs()
        assert ingestion_diff.new_docs == []
        assert ingestion_diff.deleted_docs == []
        assert ingestion_diff.updated_docs == ["source/test.md"]
        assert ingestion_diff.unchanged_docs == ["source/test2.md"]

        Path.unlink(file_path)
        ingestion_diff = ingestion_service.ingest_docs()
        assert ingestion_diff.new_docs == []
        assert ingestion_diff.deleted_docs == ["source/test.md"]
        assert ingestion_diff.updated_docs == []
        assert ingestion_diff.unchanged_docs == ["source/test2.md"]

    def test_ingest_docs_chunking_config_change_marks_all_as_updated(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Changing chunking config should mark all docs as updated."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "test.md").write_text("Some content.", encoding="utf-8")

        monkeypatch.setattr(ingestion, "BASE_DIR", tmp_path)

        chunking_config = ChunkingConfig(chunk_size=100, chunk_separator="\n\n")
        ingestion_service = IngestionService(
            chunking_config=chunking_config,
            source_dir=source_dir,
            dataset="test",
            embedding_model="test",
            db_path=tmp_path / "test.db",
        )

        ingestion_diff = ingestion_service.ingest_docs()
        with Session(ingestion_service.engine) as session:
            session.add(ingestion_diff.source_by_path["source/test.md"])
            session.commit()

        ingestion_service.chunking_config = ChunkingConfig(
            chunk_size=50, chunk_separator=". "
        )
        ingestion_diff = ingestion_service.ingest_docs()
        assert ingestion_diff.updated_docs == ["source/test.md"]
        assert ingestion_diff.new_docs == []
        assert ingestion_diff.unchanged_docs == []
