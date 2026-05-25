class LlmError(Exception):
    """Base error for LLM-related failures."""

    # Whether a caller should retry (typically with backoff). Subclasses
    # override. Defaults to False — unknown errors aren't safe to retry blindly.
    retryable: bool = False


class LlmAuthenticationError(LlmError):
    """Error raised for authentication failures with the LLM service."""

    retryable = False


class LlmRateLimitError(LlmError):
    """Error raised when rate limits are exceeded for the LLM service."""

    retryable = True


class LlmUnavailableError(LlmError):
    """Error raised when the LLM service is unavailable."""

    retryable = True


class LlmInvalidRequestError(LlmError):
    """Error raised for invalid requests to the LLM service."""

    retryable = False
