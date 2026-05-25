from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llm_lab.api.exceptions import CustomException
from llm_lab.api.middleware import LoggingMiddleware
from llm_lab.api.routers import echo, health, query
from llm_lab.llm.errors import (
    LlmAuthenticationError,
    LlmError,
    LlmInvalidRequestError,
    LlmRateLimitError,
    LlmUnavailableError,
)
from llm_lab.observability.setup import setup_logging

app = FastAPI(title="llm_lab", version="0.0.1")
setup_logging()


@app.exception_handler(CustomException)
async def custom_exception_handler(
    request: Request, exc: CustomException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
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
