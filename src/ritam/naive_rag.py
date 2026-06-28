import sys
import uuid
from typing import Annotated

import typer

from ritam.config.settings import get_settings
from ritam.core.factories import (
    create_document_source_client,
    create_llm_client,
    create_vector_store_client,
)
from ritam.core.ingestion_service import IngestionService
from ritam.core.rag_service import RagService
from ritam.llm.errors import (
    LlmAuthenticationError,
    LlmError,
    LlmInvalidRequestError,
    LlmRateLimitError,
    LlmUnavailableError,
)
from ritam.observability.context import request_id_context_var
from ritam.observability.setup import setup_logging
from ritam.retrieval.retriever import Retriever
from ritam.retrieval.types import ChunkingConfig
from ritam.vector_store.errors import VectorStoreError

app = typer.Typer()

DEFAULT_MAX_CHUNKS_PER_FILE = 1000


def take_user_input() -> str:
    try:
        user_input = input("Enter the question:\n").strip()
    except (EOFError, KeyboardInterrupt) as err:
        raise ValueError(f"User interrupted the input, exiting. {err}") from err
    if not user_input:
        raise ValueError("User input is empty, exiting...")
    return user_input


@app.command()
def index(
    dataset: Annotated[str, typer.Option(help="Dataset to index")],
    chunk_size: Annotated[int, typer.Option(help="Chunk size in characters")] = 2500,
    chunk_separator: Annotated[
        str, typer.Option(help="Chunk separator string")
    ] = "\n\n",
) -> None:
    typer.echo(
        f"Indexing dataset '{dataset}' with chunk size {chunk_size} and separator '{chunk_separator}'..."
    )
    settings = get_settings()
    llm_client = create_llm_client()
    chunking_config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_separator=chunk_separator,
    )
    document_source = create_document_source_client()
    ingestion_service = IngestionService(
        chunking_config,
        dataset,
        settings.llm_embedding_model,
        document_source=document_source,
    )
    result = ingestion_service.process_docs(llm_client, create_vector_store_client())
    typer.echo(
        f"Ingestion complete: {result.new_docs} new, {result.updated_docs} updated, "
        f"{result.unchanged_docs} unchanged, {result.deleted_docs} deleted, "
        f"{result.embedded_chunks} chunks embedded."
    )


@app.command()
def query(
    dataset: Annotated[str, typer.Option(help="Dataset to query")],
) -> None:
    typer.echo("Loading the index...")
    settings = get_settings()
    llm_client = create_llm_client()
    retriever = Retriever(
        llm_client, create_vector_store_client(), settings.similarity_threshold
    )
    rag_service = RagService(llm_client, retriever)
    query_text = take_user_input()
    result = rag_service.answer_question(
        dataset=dataset,
        embedding_model=settings.llm_embedding_model,
        query=query_text,
        top_k=3,
    )
    typer.echo("\nSources used:")
    for sc in result.chunks:
        typer.echo(f"- {sc.indexed_chunk.source} (chunk {sc.indexed_chunk.chunk_id})")
    typer.echo(f"\n\nResponse: {result.answer}")


def main() -> int:
    try:
        setup_logging()
        request_id_context_var.set(f"cli-{uuid.uuid4()}")
        app()
    except (ValueError, OSError) as err:
        typer.echo(f"Error: {err}", err=True)
        return 1
    except LlmRateLimitError as err:
        typer.echo(f"LLM Rate Limited Error: {err}", err=True)
        return 2
    except LlmAuthenticationError as err:
        typer.echo(f"LLM Authentication Error: {err}", err=True)
        return 3
    except LlmInvalidRequestError as err:
        typer.echo(f"LLM Invalid Request Error: {err}", err=True)
        return 4
    except LlmUnavailableError as err:
        typer.echo(f"LLM Unavailable Error: {err}", err=True)
        return 5
    except LlmError as err:
        typer.echo(f"LLM Error: {err}", err=True)
        return 6
    except VectorStoreError as err:
        typer.echo(f"Vector Store Error: {err}", err=True)
        return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
