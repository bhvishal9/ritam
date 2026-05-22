import csv
import json
import sys
from collections import Counter
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_lab.config.settings import get_settings
from llm_lab.core.factories import create_llm_client, create_vector_store_client
from llm_lab.core.rag_service import RagService
from llm_lab.llm.errors import (
    LlmAuthenticationError,
    LlmError,
    LlmRateLimitError,
)
from llm_lab.retrieval.retriever import Retriever


class EvalInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    dataset: str
    query: str
    expected_docs: list[str]
    query_type: str
    top_k: int | None = Field(default=None, ge=1)


class EvalOutputConfig(EvalInputConfig):
    top_k: int
    matched: bool
    num_returned: int
    returned_docs: list[str]
    returned_scores: list[float]
    error: str | None


app = typer.Typer()


def _is_matched(
    query_type: str, expected_docs: list[str], returned_docs: list[str]
) -> bool:
    """Determine whether a result is a match based on query type.

    - factual: at least one expected doc appears in returned docs
    - multi_hop: all expected docs must appear in returned docs
    - out_of_scope: matched when nothing is returned (system correctly abstained)
    """
    if query_type == "out_of_scope":
        return len(returned_docs) == 0
    if query_type == "multi_hop":
        return all(doc in returned_docs for doc in expected_docs)
    return any(doc in returned_docs for doc in expected_docs)


def load_dataset_json(path: Path) -> list[EvalInputConfig]:
    try:
        file_content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as err:
        raise ValueError(f"File {path} not found: {err}") from err
    except OSError as err:
        raise ValueError(f"Failed to read {path}: {err}") from err
    if not file_content:
        raise ValueError(f"File {path} empty")
    eval_input_config = []
    lines = file_content.split("\n")
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except JSONDecodeError as err:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_no}: {err.msg}"
            ) from err
        try:
            eval_input_config.append(EvalInputConfig(**record))
        except ValidationError as err:
            raise ValueError(
                f"Invalid dataset record in {path} at line {line_no}: {err}"
            ) from err
    if not eval_input_config:
        raise ValueError(f"File {path} has no valid examples")
    return eval_input_config


def generate_eval_output(
    example: EvalInputConfig,
    rag_service: RagService,
    top_k: int,
    embedding_model: str,
) -> EvalOutputConfig:
    try:
        result = rag_service.answer_question(
            dataset=example.dataset,
            embedding_model=embedding_model,
            query=example.query,
            top_k=top_k,
        )
    except LlmRateLimitError as e:
        typer.echo(f"Rate limit hit on {example.id}: {e}, skipping", err=True)
        return EvalOutputConfig(
            id=example.id,
            dataset=example.dataset,
            query=example.query,
            expected_docs=example.expected_docs,
            query_type=example.query_type,
            matched=False,
            num_returned=0,
            returned_docs=[],
            returned_scores=[],
            top_k=top_k,
            error="rate_limit",
        )
    except LlmAuthenticationError:
        typer.echo(f"Authentication failure on {example.id}, aborting eval.", err=True)
        raise
    except LlmError as e:
        typer.echo(f"Query failed for {example.id}: {e}, skipping", err=True)
        return EvalOutputConfig(
            id=example.id,
            dataset=example.dataset,
            query=example.query,
            expected_docs=example.expected_docs,
            query_type=example.query_type,
            matched=False,
            num_returned=0,
            returned_docs=[],
            returned_scores=[],
            top_k=top_k,
            error=e.__class__.__name__,
        )

    doc_paths = [sc.indexed_chunk.doc_path for sc in result.chunks]
    scores = [round(sc.score, 4) for sc in result.chunks]
    matched = _is_matched(example.query_type, example.expected_docs, doc_paths)
    return EvalOutputConfig(
        id=example.id,
        dataset=example.dataset,
        query=example.query,
        expected_docs=example.expected_docs,
        query_type=example.query_type,
        matched=matched,
        num_returned=len(doc_paths),
        returned_docs=doc_paths,
        returned_scores=scores,
        top_k=top_k,
        error=None,
    )


def save_eval_output(
    eval_output: list[EvalOutputConfig],
) -> None:
    output_dir = Path(__file__).parent
    results_json = output_dir / "results.json"
    results_csv = output_dir / "results.csv"
    try:
        data = [output.model_dump() for output in eval_output]
        results_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        field_names = sorted({k for r in data for k in r})
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=field_names, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
    except Exception as err:
        raise ValueError(f"Error saving results: {err}") from err


def print_eval_output(default_top_k: int, eval_output: list[EvalOutputConfig]) -> None:
    output_dir = Path(__file__).parent
    results_json = output_dir / "results.json"
    results_csv = output_dir / "results.csv"
    total_examples = len(eval_output)
    errored_examples = sum(1 for output in eval_output if output.error is not None)
    error_breakdown = Counter(
        output.error for output in eval_output if output.error is not None
    )
    top_k_values = sorted({output.top_k for output in eval_output})
    top_k_label = str(default_top_k)
    if top_k_values:
        top_k_label = (
            str(top_k_values[0])
            if len(top_k_values) == 1
            else f"mixed values {top_k_values} (default={default_top_k})"
        )

    # Recall metrics — factual and multi_hop only
    retrieval_examples = [
        o for o in eval_output if o.query_type in ("factual", "multi_hop")
    ]
    retrieval_total = len(retrieval_examples)
    retrieval_matched = sum(1 for o in retrieval_examples if o.matched)
    retrieval_errored = sum(1 for o in retrieval_examples if o.error is not None)
    retrieval_processed = retrieval_total - retrieval_errored
    retrieval_returned = sum(
        1 for o in retrieval_examples if o.error is None and o.num_returned > 0
    )
    observed_recall = (
        retrieval_matched / retrieval_total if retrieval_total > 0 else 0.0
    )
    retrieval_recall = (
        retrieval_matched / retrieval_processed if retrieval_processed > 0 else 0.0
    )
    coverage = (
        retrieval_returned / retrieval_processed if retrieval_processed > 0 else 0.0
    )

    # Abstention metric — out_of_scope only
    oos_examples = [o for o in eval_output if o.query_type == "out_of_scope"]
    oos_total = len(oos_examples)
    oos_matched = sum(1 for o in oos_examples if o.matched)
    oos_errored = sum(1 for o in oos_examples if o.error is not None)
    oos_processed = oos_total - oos_errored
    abstention_rate = oos_matched / oos_processed if oos_processed > 0 else 0.0

    typer.echo(f"Total examples: {total_examples}")
    typer.echo(f"Errored examples: {errored_examples}")
    typer.echo("")
    typer.echo(f"Retrieval examples (factual + multi_hop): {retrieval_total}")
    typer.echo(f"  Matched: {retrieval_matched}")
    typer.echo(f"  Missed: {retrieval_processed - retrieval_matched}")
    typer.echo(
        f"  Observed recall at {top_k_label} (matched/total): {observed_recall:.3f}"
    )
    typer.echo(
        f"  Retrieval recall at {top_k_label} (matched/non-error): {retrieval_recall:.3f}"
    )
    typer.echo(
        f"  Coverage at {top_k_label} (returned>0 among non-error): {coverage:.3f}"
    )
    typer.echo("")
    typer.echo(f"Out-of-scope examples: {oos_total}")
    typer.echo(f"  Abstention rate (correctly returned nothing): {abstention_rate:.3f}")

    query_types = sorted({o.query_type for o in eval_output})
    if len(query_types) > 1:
        typer.echo("Breakdown by query type:")
        for qt in query_types:
            qt_results = [o for o in eval_output if o.query_type == qt]
            qt_total = len(qt_results)
            qt_matched = sum(1 for o in qt_results if o.matched)
            qt_errored = sum(1 for o in qt_results if o.error is not None)
            qt_processed = qt_total - qt_errored
            qt_recall = qt_matched / qt_processed if qt_processed > 0 else 0.0
            typer.echo(
                f"  {qt}: {qt_matched}/{qt_processed} matched (recall={qt_recall:.3f}, errors={qt_errored})"
            )

    if error_breakdown:
        typer.echo("Error breakdown:")
        for error_name, count in sorted(error_breakdown.items()):
            typer.echo(f"  {error_name}: {count}")
    typer.echo(f"Wrote results to {results_csv.name} and {results_json.name}")


@app.command()
def run_eval(
    top_k: Annotated[int, typer.Option(help="Default top_k value")] = 3,
    dataset_file: Annotated[Path, typer.Option(help="Path to dataset file")] = Path(
        "dataset.jsonl"
    ),
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    typer.echo("Running eval...")
    input_file = dataset_file.expanduser()
    if not input_file.is_absolute():
        input_file = Path(__file__).parent / input_file
    eval_input_config = load_dataset_json(input_file)
    settings = get_settings()
    llm_client = create_llm_client()
    rag_service = RagService(
        llm_client,
        Retriever(llm_client, create_vector_store_client()),
    )
    eval_output_config = []
    for config in eval_input_config:
        example_top_k = config.top_k if config.top_k is not None else top_k
        eval_output = generate_eval_output(
            config, rag_service, example_top_k, settings.llm_embedding_model
        )
        eval_output_config.append(eval_output)

    save_eval_output(eval_output_config)
    print_eval_output(top_k, eval_output_config)


def main() -> int:
    try:
        app()
    except (ValueError, OSError) as err:
        typer.echo(f"Error: {err}", err=True)
        return 1
    except LlmAuthenticationError as err:
        typer.echo(f"LLM Authentication Error: {err}", err=True)
        return 2
    except LlmError as err:
        typer.echo(f"LLM Error: {err}", err=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
