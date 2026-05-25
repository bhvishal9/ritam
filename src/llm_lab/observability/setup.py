import logging

from llm_lab.observability.logging import JsonFormatter

NOISY_LIBRARIES = ("httpx", "httpcore", "google_genai", "watchfiles", "uvicorn.access")


_CONFIGURED = False


def setup_logging() -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: safe to call multiple times within a process. We replace any
    pre-existing handlers (e.g. uvicorn's default text handler) rather than
    bailing out, so the JSON format is consistent across the whole process.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    _CONFIGURED = True
