import json
import logging
from datetime import UTC, datetime

from llm_lab.observability.context import (
    candidate_k_context_var,
    chunks_return_context_var,
    dataset_context_var,
    embed_ms_context_var,
    generate_ms_context_var,
    request_id_context_var,
    retrieve_ms_context_var,
    stage_context_var,
    top_k_context_var,
)


def _snapshot_contextvars() -> dict[str, object]:
    raw = {
        "candidate_k": candidate_k_context_var.get(),
        "chunks_returned": chunks_return_context_var.get(),
        "dataset": dataset_context_var.get(),
        "embed_ms": embed_ms_context_var.get(),
        "generate_ms": generate_ms_context_var.get(),
        "request_id": request_id_context_var.get(),
        "retrieve_ms": retrieve_ms_context_var.get(),
        "stage": stage_context_var.get(),
        "top_k": top_k_context_var.get(),
    }
    return {k: v for k, v in raw.items() if v is not None and v != "not-set"}


class JsonFormatter(logging.Formatter):
    """Custom formatter for logging to JSON format."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        payload.update(_snapshot_contextvars())
        return json.dumps(payload)
