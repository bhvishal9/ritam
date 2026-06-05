from typing import Protocol

from pydantic import BaseModel


class DocumentSourceOutput(BaseModel):
    doc_path: str
    doc_content: str


class DocumentSource(Protocol):
    """Protocol describing the interface for document sources."""

    def load(self, dataset: str) -> list[DocumentSourceOutput]:
        """Load documents for the given dataset and return a list of DocumentStoreOutput objects."""
        ...
