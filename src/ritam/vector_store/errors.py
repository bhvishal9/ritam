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


class IndexSchemaMismatchError(VectorStoreError):
    """Raised when an existing collection does not match the expected schema.

    Creating a collection is idempotent, which silently becomes a no-op when the
    desired schema has changed since the collection was built. Writing the new
    shape into the old collection would fail later with an opaque error, so
    detect the mismatch up front and tell the operator what to do.
    """
