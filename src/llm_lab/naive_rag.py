import sys
from pathlib import Path
from typing import Annotated

import typer

from llm_lab.config.paths import DEFAULT_DOCS_DIR
from llm_lab.config.settings import get_settings
from llm_lab.core.factories import create_llm_client, create_vector_store_client
from llm_lab.core.rag_service import RagService
from llm_lab.llm.errors import (
    LlmAuthenticationError,
    LlmError,
    LlmInvalidRequestError,
    LlmRateLimitError,
    LlmUnavailableError,
)
from llm_lab.retrieval.indexing import Indexer
from llm_lab.retrieval.retriever import Retriever
from llm_lab.retrieval.types import ChunkingConfig

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
    source_dir: Annotated[
        Path, typer.Option(help="Source directory")
    ] = DEFAULT_DOCS_DIR,
    chunk_size: Annotated[int, typer.Option(help="Chunk size in characters")] = 10000,
    chunk_separator: Annotated[
        str, typer.Option(help="Chunk separator string")
    ] = "\n\n",
) -> None:
    typer.echo(f"Indexing dataset '{dataset}' from {source_dir}")
    settings = get_settings()
    llm_client = create_llm_client()
    chunking_config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_separator=chunk_separator,
    )
    indexer = Indexer(
        source_dir, settings.llm_embedding_model, dataset, chunking_config
    )
    indexed_chunks, docs_count = indexer.run(llm_client)
    vector_store_client = create_vector_store_client()
    vector_store_client.store(
        indexed_chunks, dataset, settings.llm_embedding_model, docs_count
    )


@app.command()
def query(
    dataset: Annotated[str, typer.Option(help="Dataset to query")],
) -> None:
    typer.echo("Loading the index...")
    settings = get_settings()
    llm_client = create_llm_client()
    retriever = Retriever(llm_client, create_vector_store_client())
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
