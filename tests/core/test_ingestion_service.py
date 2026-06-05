from pathlib import Path

from sqlmodel import Session

from llm_lab.core.ingestion_service import IngestionService
from llm_lab.document_source.local_document_source import LocalDocumentSource
from llm_lab.retrieval.types import ChunkingConfig


class TestIngestionService:
    def test_ingest_docs(self, tmp_path: Path) -> None:
        """Test ingestion service's ingest_docs method."""
        dataset_dir = tmp_path / "test"
        dataset_dir.mkdir()
        file_path = dataset_dir / "test.md"
        file_path.write_text(
            "This is a test document. It will be indexed.", encoding="utf-8"
        )
        file_2_path = dataset_dir / "test2.md"
        file_2_path.write_text(
            "This is a test document. It will be indexed.", encoding="utf-8"
        )
        chunk_size = 70
        chunk_separator = ". "
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size, chunk_separator=chunk_separator
        )

        document_source = LocalDocumentSource(tmp_path.as_uri())
        ingestion_service = IngestionService(
            chunking_config=chunking_config,
            dataset="test",
            embedding_model="test",
            document_source=document_source,
            db_path=tmp_path / "test.db",
        )
        ingestion_plan = ingestion_service.ingest_docs()
        assert set(ingestion_plan.new_docs) == {"test2.md", "test.md"}
        assert ingestion_plan.deleted_docs == []
        assert ingestion_plan.updated_docs == []
        assert ingestion_plan.unchanged_docs == []

        with Session(ingestion_service.engine) as session:
            session.add(ingestion_plan.source_by_path["test.md"])
            session.add(ingestion_plan.source_by_path["test2.md"])
            session.commit()

        ingestion_plan = ingestion_service.ingest_docs()
        assert ingestion_plan.new_docs == []
        assert ingestion_plan.deleted_docs == []
        assert ingestion_plan.updated_docs == []
        assert set(ingestion_plan.unchanged_docs) == {"test2.md", "test.md"}

        file_path.write_text(
            "This is a test document. It will be indexed again.", encoding="utf-8"
        )
        ingestion_plan = ingestion_service.ingest_docs()
        assert ingestion_plan.new_docs == []
        assert ingestion_plan.deleted_docs == []
        assert ingestion_plan.updated_docs == ["test.md"]
        assert ingestion_plan.unchanged_docs == ["test2.md"]

        Path.unlink(file_path)
        ingestion_plan = ingestion_service.ingest_docs()
        assert ingestion_plan.new_docs == []
        assert ingestion_plan.deleted_docs == ["test.md"]
        assert ingestion_plan.updated_docs == []
        assert ingestion_plan.unchanged_docs == ["test2.md"]

    def test_ingest_docs_chunking_config_change_marks_all_as_updated(
        self, tmp_path: Path
    ) -> None:
        """Changing chunking config should mark all docs as updated."""
        dataset_dir = tmp_path / "test"
        dataset_dir.mkdir()
        (dataset_dir / "test.md").write_text("Some content.", encoding="utf-8")

        chunking_config = ChunkingConfig(chunk_size=100, chunk_separator="\n\n")
        document_source = LocalDocumentSource(tmp_path.as_uri())
        ingestion_service = IngestionService(
            chunking_config=chunking_config,
            dataset="test",
            embedding_model="test",
            document_source=document_source,
            db_path=tmp_path / "test.db",
        )

        ingestion_plan = ingestion_service.ingest_docs()
        with Session(ingestion_service.engine) as session:
            session.add(ingestion_plan.source_by_path["test.md"])
            session.commit()

        ingestion_service.chunking_config = ChunkingConfig(
            chunk_size=50, chunk_separator=". "
        )
        ingestion_plan = ingestion_service.ingest_docs()
        assert ingestion_plan.updated_docs == ["test.md"]
        assert ingestion_plan.new_docs == []
        assert ingestion_plan.unchanged_docs == []
