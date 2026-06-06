import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, create_engine, select

from ritam.config.paths import DEFAULT_SQLITE_DB_PATH
from ritam.document_source.types import DocumentSource
from ritam.llm.types import LlmClient
from ritam.observability.context import stage
from ritam.retrieval.indexing import Indexer
from ritam.retrieval.types import ChunkingConfig, IndexerInput
from ritam.vector_store.types import VectorStoreClient


class IngestionSchema(SQLModel, table=True):
    doc_path: str = Field(primary_key=True)
    dataset: str = Field(primary_key=True)
    embedding_model: str = Field(primary_key=True)
    index_fingerprint: str


@dataclass
class IngestionPlan:
    new_docs: list[str]
    updated_docs: list[str]
    unchanged_docs: list[str]
    deleted_docs: list[str]
    source_by_path: dict[str, IngestionSchema]
    content_by_path: dict[str, str]


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


class IngestionService:
    def __init__(
        self,
        chunking_config: ChunkingConfig,
        dataset: str,
        embedding_model: str,
        document_source: DocumentSource,
        db_path: Path = DEFAULT_SQLITE_DB_PATH,
    ):
        self.chunking_config = chunking_config
        self.db_path = db_path
        self.engine = create_db_and_tables(db_path=db_path)
        self.dataset = dataset
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        self.document_source = document_source

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

    def ingest_docs(self) -> IngestionPlan:
        """Ingest all Markdown files in the source directory."""
        source_records: list[IngestionSchema] = []
        content_by_path = {}
        current_records = self.get_current_records()
        source_docs = self.document_source.load(self.dataset)
        for doc in source_docs:
            doc_path = doc.doc_path
            doc_content = doc.doc_content
            hash_input = f"{self.chunking_config.chunk_size}-{self.chunking_config.chunk_separator}-{doc_content}"
            index_fingerprint = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
            source_records.append(
                IngestionSchema(
                    doc_path=doc_path,
                    dataset=self.dataset,
                    embedding_model=self.embedding_model,
                    index_fingerprint=index_fingerprint,
                )
            )
            content_by_path[doc_path] = doc_content
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
        return IngestionPlan(
            new_docs=new_docs,
            updated_docs=updated_docs,
            unchanged_docs=unchanged_docs,
            deleted_docs=deleted_docs,
            source_by_path=source_by_path,
            content_by_path=content_by_path,
        )

    def process_docs(
        self, llm_client: LlmClient, vector_store_client: VectorStoreClient
    ) -> IngestionResult:
        """Process all the documents in the source directory."""
        self.logger.info(
            "ingest_start",
            extra={
                "fields": {
                    "dataset": self.dataset,
                    "embedding_model": self.embedding_model,
                }
            },
        )
        ingestion_plan = self.ingest_docs()
        self.logger.info(
            "ingest_diff",
            extra={
                "fields": {
                    "new": len(ingestion_plan.new_docs),
                    "updated": len(ingestion_plan.updated_docs),
                    "unchanged": len(ingestion_plan.unchanged_docs),
                    "deleted": len(ingestion_plan.deleted_docs),
                }
            },
        )
        indexer = Indexer(self.embedding_model, self.chunking_config)
        indexing_docs = ingestion_plan.new_docs + ingestion_plan.updated_docs
        docs_to_index = [
            IndexerInput(doc_path=doc, doc_content=ingestion_plan.content_by_path[doc])
            for doc in indexing_docs
        ]
        with stage("index"):
            chunks = indexer.build_index(llm_client, docs_to_index)
        for doc in ingestion_plan.updated_docs + ingestion_plan.deleted_docs:
            vector_store_client.delete(self.dataset, self.embedding_model, str(doc))
        if len(chunks) > 0:
            vector_store_client.store(
                chunks, self.dataset, self.embedding_model, len(chunks)
            )
        with Session(self.engine) as session:
            for doc in ingestion_plan.new_docs:
                session.add(ingestion_plan.source_by_path[doc])
            for doc in ingestion_plan.updated_docs:
                result = select(IngestionSchema).where(
                    IngestionSchema.dataset == self.dataset,
                    IngestionSchema.embedding_model == self.embedding_model,
                    IngestionSchema.doc_path == doc,
                )
                record = session.exec(result).one()
                record.index_fingerprint = ingestion_plan.source_by_path[
                    doc
                ].index_fingerprint
                session.add(record)
            for doc in ingestion_plan.deleted_docs:
                result = select(IngestionSchema).where(
                    IngestionSchema.dataset == self.dataset,
                    IngestionSchema.embedding_model == self.embedding_model,
                    IngestionSchema.doc_path == doc,
                )
                record = session.exec(result).one()
                session.delete(record)
            session.commit()
        ingest_result = IngestionResult(
            new_docs=len(ingestion_plan.new_docs),
            updated_docs=len(ingestion_plan.updated_docs),
            unchanged_docs=len(ingestion_plan.unchanged_docs),
            deleted_docs=len(ingestion_plan.deleted_docs),
            embedded_chunks=len(chunks),
        )
        self.logger.info(
            "ingest_complete",
            extra={
                "fields": {
                    "dataset": self.dataset,
                    "embedding_model": self.embedding_model,
                    "new_docs": ingest_result.new_docs,
                    "updated_docs": ingest_result.updated_docs,
                    "unchanged_docs": ingest_result.unchanged_docs,
                    "deleted_docs": ingest_result.deleted_docs,
                    "embedded_chunks": ingest_result.embedded_chunks,
                }
            },
        )
        return ingest_result
