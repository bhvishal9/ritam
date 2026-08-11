"""Per-request observability fields.

Values live in a single mutable dict held by one ContextVar, rather than in one
ContextVar per field. This matters because request handlers may run off the
event loop: FastAPI executes a plain ``def`` endpoint on a worker thread, and
crossing that boundary *copies* the context. Values ``set()`` inside the copy are
discarded when it goes away, so a per-field ContextVar written inside the handler
would be invisible to middleware running afterwards on the loop.

Copying a context copies the *bindings*, so both sides end up pointing at the
same dict object. Mutating that dict is therefore visible on both sides of a
thread hop, while the per-request isolation that makes contextvars safe under
concurrency is preserved — each request binds its own dict.

``RequestField`` keeps the ContextVar API (``get`` / ``set`` / ``reset``) so call
sites read the same as before.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

_request_fields: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_fields", default=None
)


class _Missing:
    """Sentinel: the field had no value before it was set."""


_MISSING = _Missing()


def _store_for_write() -> dict[str, Any]:
    """Return the current field store, creating and binding one if absent.

    Lazy creation keeps non-HTTP entrypoints (CLI, tests) working without
    ceremony. The HTTP path does not rely on it — ``bind_request_fields`` runs in
    middleware, on the event loop, before any thread hop — and neither does the
    eval runner, which binds via ``request_scope``.
    """
    store = _request_fields.get()
    if store is None:
        store = {}
        _request_fields.set(store)
    return store


def _store_for_read() -> dict[str, Any]:
    """Return the current field store, or an empty one, without binding.

    Reads must never bind. A binding created by a read leaks upward into
    whatever context happens to be active — and because ``copy_context()``
    copies the *binding*, every later copy would then share that one dict.
    A single log line emitted before a loop would silently join every iteration
    of that loop into one store.
    """
    return _request_fields.get() or {}


@contextmanager
def bind_request_fields() -> Iterator[dict[str, Any]]:
    """Bind a fresh field store for one request, and yield it.

    Must be entered before the handler runs, on whatever thread will later read
    the values back — otherwise the binding is made in a context copy and is
    lost. Yields the dict so the caller can read the accumulated fields after
    the handler returns.
    """
    store: dict[str, Any] = {}
    token = _request_fields.set(store)
    try:
        yield store
    finally:
        _request_fields.reset(token)


@dataclass(frozen=True)
class FieldToken[T]:
    """Restores a field to the value it held before a ``set``."""

    name: str
    previous: T | _Missing


class RequestField[T]:
    """A single observability field, stored in the shared per-request dict."""

    def __init__(self, name: str, default: T) -> None:
        self._name = name
        self._default = default

    @property
    def name(self) -> str:
        return self._name

    def get(self) -> T:
        store = _store_for_read()
        if self._name not in store:
            return self._default
        return cast(T, store[self._name])

    def set(self, value: T) -> FieldToken[T]:
        store = _store_for_write()
        previous: T | _Missing = store.get(self._name, _MISSING)
        store[self._name] = value
        return FieldToken(name=self._name, previous=previous)

    def reset(self, token: FieldToken[T]) -> None:
        store = _store_for_write()
        if isinstance(token.previous, _Missing):
            store.pop(token.name, None)
        else:
            store[token.name] = token.previous


dataset_context_var: RequestField[str | None] = RequestField("dataset", None)
stage_context_var: RequestField[str | None] = RequestField("stage", None)
top_k_context_var: RequestField[int | None] = RequestField("top_k", None)
candidate_k_context_var: RequestField[int | None] = RequestField("candidate_k", None)
request_id_context_var: RequestField[str] = RequestField("request_id", "not-set")
embed_ms_context_var: RequestField[float | None] = RequestField("embed_ms", None)
retrieve_ms_context_var: RequestField[float | None] = RequestField("retrieve_ms", None)
generate_ms_context_var: RequestField[float | None] = RequestField("generate_ms", None)
chunks_return_context_var: RequestField[int | None] = RequestField(
    "chunks_returned", None
)
input_tokens_context_var: RequestField[int | None] = RequestField("input_tokens", None)
output_tokens_context_var: RequestField[int | None] = RequestField(
    "output_tokens", None
)
input_cost_usd_context_var: RequestField[float | None] = RequestField(
    "input_cost_usd", None
)
output_cost_usd_context_var: RequestField[float | None] = RequestField(
    "output_cost_usd", None
)
total_cost_usd_context_var: RequestField[float | None] = RequestField(
    "total_cost_usd", None
)
model_context_var: RequestField[str | None] = RequestField("model", None)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Mark the current pipeline stage for logs emitted inside this block.

    Any log line emitted while this context is active will carry
    ``stage=<name>`` via the formatter's field snapshot. Restores the prior
    value on exit (supports nested stages, though we don't currently use that).
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
    """Scope the request-level fields (request_id, dataset, top_k).

    Use for non-HTTP entrypoints (eval runner, batch jobs) where there's no
    middleware to manage the request lifecycle.

    Binds a *fresh* field store, so every field — not just the three set here —
    starts empty. Without that, an example which never reaches generation would
    report the previous example's ``generate_ms`` and cost, because nothing
    clears fields the pipeline didn't write.
    """
    with bind_request_fields():
        request_id_context_var.set(request_id)
        if dataset is not None:
            dataset_context_var.set(dataset)
        if top_k is not None:
            top_k_context_var.set(top_k)
        yield
