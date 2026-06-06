from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ritam.api.exceptions import CustomException
from ritam.api.middleware import LoggingMiddleware
from ritam.api.routers import echo, health, query
from ritam.llm.errors import (
    LlmAuthenticationError,
    LlmError,
    LlmInvalidRequestError,
    LlmRateLimitError,
    LlmUnavailableError,
)
from ritam.observability.setup import setup_logging
from ritam.vector_store.errors import IndexNotFoundError, VectorStoreError

app = FastAPI(title="ritam", version="0.0.1")
setup_logging()


@app.exception_handler(CustomException)
async def custom_exception_handler(
    request: Request, exc: CustomException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


@app.exception_handler(IndexNotFoundError)
async def index_not_found_exception_handler(
    request: Request, exc: IndexNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": str(exc)},
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(
    request: Request, exc: VectorStoreError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )


@app.exception_handler(LlmRateLimitError)
async def llm_rate_limit_exception_handler(
    request: Request, exc: LlmRateLimitError
) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": str(exc)},
    )


@app.exception_handler(LlmAuthenticationError)
async def llm_authentication_exception_handler(
    request: Request, exc: LlmAuthenticationError
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": str(exc)},
    )


@app.exception_handler(LlmInvalidRequestError)
async def llm_invalid_request_exception_handler(
    request: Request, exc: LlmInvalidRequestError
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": str(exc)},
    )


@app.exception_handler(LlmUnavailableError)
async def llm_unavailable_exception_handler(
    request: Request, exc: LlmUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": str(exc)},
    )


@app.exception_handler(LlmError)
async def llm_generic_error_exception_handler(
    request: Request, exc: LlmError
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": str(exc)},
    )


app.add_middleware(LoggingMiddleware)

app.include_router(echo.router)
app.include_router(health.router)
app.include_router(query.router)
