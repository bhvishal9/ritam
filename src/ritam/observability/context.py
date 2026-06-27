from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

dataset_context_var: ContextVar[str | None] = ContextVar("dataset", default=None)
stage_context_var: ContextVar[str | None] = ContextVar("stage", default=None)
top_k_context_var: ContextVar[int | None] = ContextVar("top_k", default=None)
candidate_k_context_var: ContextVar[int | None] = ContextVar(
    "candidate_k", default=None
)
request_id_context_var = ContextVar("request_id", default="not-set")
embed_ms_context_var: ContextVar[float | None] = ContextVar("embed_ms", default=None)
retrieve_ms_context_var: ContextVar[float | None] = ContextVar(
    "retrieve_ms", default=None
)
generate_ms_context_var: ContextVar[float | None] = ContextVar(
    "generate_ms", default=None
)
chunks_return_context_var: ContextVar[int | None] = ContextVar(
    "chunks_returned", default=None
)
input_tokens_context_var: ContextVar[int | None] = ContextVar(
    "input_tokens", default=None
)
output_tokens_context_var: ContextVar[int | None] = ContextVar(
    "output_tokens", default=None
)
input_cost_usd_context_var: ContextVar[float | None] = ContextVar(
    "input_cost_usd", default=None
)
output_cost_usd_context_var: ContextVar[float | None] = ContextVar(
    "output_cost_usd", default=None
)
total_cost_usd_context_var: ContextVar[float | None] = ContextVar(
    "total_cost_usd", default=None
)
model_context_var: ContextVar[str | None] = ContextVar("model", default=None)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Mark the current pipeline stage for logs emitted inside this block.

    Any log line emitted while this context is active will carry
    ``stage=<name>`` via the formatter's contextvar snapshot. Restores the
    prior value on exit (supports nested stages, though we don't currently
    use that).
    """
    token = stage_context_var.set(name)
    try:
        yield
    finally:
        stage_context_var.reset(token)


@contextmanager
def request_scope(
    request_id: str, dataset: str | None = None, top_k: int | None = None
) -> Iterator[None]:
    """Scope the request-level contextvars (request_id, dataset, top_k).

    Use for non-HTTP entrypoints (eval runner, batch jobs) where there's no
    middleware to manage the request lifecycle. Sets the vars on entry and
    restores prior values on exit, so logs emitted between scopes don't
    carry stale values from the previous example.
    """
    resets: list[tuple[ContextVar[Any], Token[Any]]] = [
        (request_id_context_var, request_id_context_var.set(request_id))
    ]
    if dataset is not None:
        resets.append((dataset_context_var, dataset_context_var.set(dataset)))
    if top_k is not None:
        resets.append((top_k_context_var, top_k_context_var.set(top_k)))
    try:
        yield
    finally:
        # Reset in reverse order so nested scopes restore correctly.
        for var, token in reversed(resets):
            var.reset(token)
