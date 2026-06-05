class VectorStoreError(Exception):
    """Base error for vector store failures."""


class IndexNotFoundError(VectorStoreError):
    """Raised when no index exists for the requested embedding model.

    This is a caller-facing condition (the dataset was never indexed with this
    embedding model), not a server fault — map it to a 4xx at the edge.
    """


class VectorStorePayloadError(VectorStoreError):
    """Raised when a stored point is missing its expected payload.

    This indicates corrupt or incomplete index state — a server-side fault.
    """
