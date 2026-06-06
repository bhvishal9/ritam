from datetime import datetime

from pydantic import BaseModel, Field

from ritam.vector_store.types import IndexedChunk


class IndexFile(BaseModel):
    index_id: str = Field(description="A unique identifier for this index file.")
    chunks: list[IndexedChunk] = Field(
        description="A list of indexed chunks contained in this file."
    )


class ManifestIndexFile(BaseModel):
    index_id: str = Field(
        description='Unique identifier for the index (e.g., "index-0001").'
    )
    path: str = Field(
        description='Relative path to the index file (e.g., "index-0001.json").'
    )
    num_chunks: int = Field(
        description="Number of chunks contained within this index file."
    )


class ManifestFile(BaseModel):
    dataset: str = Field(
        description="The name of the dataset to which this manifest belongs, as passed to the CLI."
    )
    embedding_model: str = Field(
        description="The embedding model used for the dataset documented in this manifest."
    )
    created_at: datetime = Field(
        description="The timestamp when this manifest file was created."
    )
    total_docs: int = Field(
        description="The total number of documents recorded in this manifest."
    )
    total_chunks: int = Field(
        description="The total number of chunks across all documents and index files in this manifest."
    )
    index_files: list[ManifestIndexFile] = Field(
        description="A list of index file entries, each detailing an index shard."
    )
