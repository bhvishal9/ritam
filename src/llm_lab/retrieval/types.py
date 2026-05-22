from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(
        description="The desired chunk size in characters for document processing.",
        gt=0,
    )
    chunk_separator: str = Field(
        description="The separator string used to delineate chunks.",
        min_length=1,
    )
