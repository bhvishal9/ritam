import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, create_engine, select

from llm_lab.config.paths import BASE_DIR, DEFAULT_SQLITE_DB_PATH
from llm_lab.core.factories import create_vector_store_client
from llm_lab.llm.types import LlmClient
from llm_lab.retrieval.indexing import Indexer
from llm_lab.retrieval.types import ChunkingConfig


class IngestionSchema(SQLModel, table=True):  # type: ignore[misc, call-arg]
    doc_path: str = Field(primary_key=True)
    dataset: str = Field(primary_key=True)
    embedding_model: str = Field(primary_key=True)
    index_fingerprint: str


@dataclass
class IngestionDiff:
    new_docs: list[str]
    updated_docs: list[str]
    unchanged_docs: list[str]
    deleted_docs: list[str]
    source_by_path: dict[str, IngestionSchema]


@dataclass
class IngestionResult:
    new_docs: int
    updated_docs: int
    unchanged_docs: int
    deleted_docs: int
    embedded_chunks: int


def create_db_and_tables(db_path: Path) -> Engine:
    """Create the database session and tables."""
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)
        return engine
    except Exception as err:
        raise RuntimeError(f"Error creating sqlite database: {err}") from err


def load_docs(source_dir: Path) -> list[Path]:
    """Load all Markdown files from the source directory."""
    if not source_dir.exists():
        raise ValueError(f"Directory {source_dir} does not exist")
    files = list(source_dir.glob("**/*.md"))
    if not files:
        raise ValueError(f"No Markdown files found in directory {source_dir}")
    return files


class IngestionService:
    def __init__(
        self,
        chunking_config: ChunkingConfig,
        source_dir: Path,
        dataset: str,
        embedding_model: str,
        db_path: Path = DEFAULT_SQLITE_DB_PATH,
    ):
        self.chunking_config = chunking_config
        self.db_path = db_path
        self.engine = create_db_and_tables(db_path=db_path)
        self.source_dir = source_dir
        self.dataset = dataset
        self.embedding_model = embedding_model

    def get_current_records(self) -> list[IngestionSchema]:
        """Get the current records from the database."""
        try:
            with Session(self.engine) as session:
                current_record_statement = select(IngestionSchema).where(
                    IngestionSchema.dataset == self.dataset,
                    IngestionSchema.embedding_model == self.embedding_model,
                )
                return cast(
                    list[IngestionSchema], session.exec(current_record_statement).all()
                )
        except Exception as err:
            raise RuntimeError(f"Error querying sqlite database: {err}") from err

    def ingest_docs(self) -> IngestionDiff:
        """Ingest all Markdown files in the source directory."""
        source_records: list[IngestionSchema] = []
        current_records = self.get_current_records()
        source_docs = load_docs(self.source_dir)
        for doc in source_docs:
            doc_path = str(doc.relative_to(BASE_DIR))
            doc_content = doc.read_text(encoding="utf-8")
            hash_input = f"{self.chunking_config.chunk_size}-{self.chunking_config.chunk_separator}-{doc_content}"
            index_fingerprint = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
            source_records.append(
                IngestionSchema(
                    index_fingerprint=index_fingerprint,
                    doc_path=doc_path,
                    dataset=self.dataset,
                    embedding_model=self.embedding_model,
                )
            )
        source_by_path = {r.doc_path: r for r in source_records}
        current_by_path = {r.doc_path: r for r in current_records}
        source_paths = {r.doc_path for r in source_records}
        current_paths = {r.doc_path for r in current_records}
        new_docs = [
            r.doc_path for r in source_records if r.doc_path not in current_paths
        ]
        updated_docs = [
            r.doc_path
            for r in source_records
            if r.doc_path in current_by_path
            and r.index_fingerprint != current_by_path[r.doc_path].index_fingerprint
        ]

        unchanged_docs = [
            r.doc_path
            for r in source_records
            if r.doc_path in current_by_path
            and r.index_fingerprint == current_by_path[r.doc_path].index_fingerprint
        ]
        deleted_docs = [
            r.doc_path for r in current_records if r.doc_path not in source_paths
        ]
        return IngestionDiff(
            new_docs=new_docs,
            updated_docs=updated_docs,
            unchanged_docs=unchanged_docs,
            deleted_docs=deleted_docs,
            source_by_path=source_by_path,
        )

    def process_docs(self, llm_client: LlmClient) -> IngestionResult:
        """Process all the documents in the source directory."""
        ingestion_diff = self.ingest_docs()
        indexer = Indexer(self.embedding_model, self.chunking_config)
        vector_store_client = create_vector_store_client()
        indexing_docs = ingestion_diff.new_docs + ingestion_diff.updated_docs
        chunks = indexer.build_index(llm_client, indexing_docs)
        for doc in ingestion_diff.updated_docs + ingestion_diff.deleted_docs:
            vector_store_client.delete(self.dataset, self.embedding_model, str(doc))
        if len(chunks) > 0:
            vector_store_client.store(
                chunks, self.dataset, self.embedding_model, len(chunks)
            )
        with Session(self.engine) as session:
            for doc in ingestion_diff.new_docs:
                session.add(ingestion_diff.source_by_path[doc])
            for doc in ingestion_diff.updated_docs:
                result = select(IngestionSchema).where(
                    IngestionSchema.dataset == self.dataset,
                    IngestionSchema.embedding_model == self.embedding_model,
                    IngestionSchema.doc_path == doc,
                )
                record = session.exec(result).one()
                record.index_fingerprint = ingestion_diff.source_by_path[
                    doc
                ].index_fingerprint
                session.add(record)
            for doc in ingestion_diff.deleted_docs:
                result = select(IngestionSchema).where(
                    IngestionSchema.dataset == self.dataset,
                    IngestionSchema.embedding_model == self.embedding_model,
                    IngestionSchema.doc_path == doc,
                )
                record = session.exec(result).one()
                session.delete(record)
            session.commit()
        return IngestionResult(
            new_docs=len(ingestion_diff.new_docs),
            updated_docs=len(ingestion_diff.updated_docs),
            unchanged_docs=len(ingestion_diff.unchanged_docs),
            deleted_docs=len(ingestion_diff.deleted_docs),
            embedded_chunks=len(chunks),
        )
