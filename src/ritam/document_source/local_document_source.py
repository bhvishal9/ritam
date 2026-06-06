import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

from ritam.document_source.types import DocumentSource, DocumentSourceOutput


class LocalDocumentSource(DocumentSource):
    def __init__(self, source_uri: str):
        self.source_uri: str = source_uri
        self.logger = logging.getLogger(__name__)

    def load(self, dataset: str) -> list[DocumentSourceOutput]:
        uri_dir = unquote(urlparse(self.source_uri).path)
        source_dir = Path(uri_dir) / dataset
        self.logger.info(f"Loading {source_dir}")
        dir_path = Path(source_dir)
        if not dir_path.is_dir():
            raise ValueError(f"Source directory {source_dir} is not a directory")
        md_files = list(dir_path.glob("**/*.md"))
        if not md_files:
            raise ValueError(f"No Markdown files found in directory {source_dir}")
        result: list[DocumentSourceOutput] = []
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                doc_path = md_file.relative_to(dir_path)
                if not content:
                    self.logger.warning(f"File {md_file} is empty, skipping...")
                    continue
                result.append(
                    DocumentSourceOutput(doc_path=str(doc_path), doc_content=content)
                )
            except FileNotFoundError as err:
                raise ValueError(
                    f"File {md_file} not found in directory {source_dir}: {err}"
                ) from err

        return result
