# --- Builder: carries the compiler toolchain to build any dep lacking a 3.14 wheel ---
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Resolve dependencies first so this layer caches independently of src changes.
COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-install-project
COPY src /app/src
RUN uv sync --frozen

# --- Runtime: slim, no compiler and no uv — exec uvicorn straight from the baked venv ---
FROM python:3.14-slim
RUN groupadd -r ritam && useradd -r -g ritam ritam --home /app
COPY --from=builder --chown=ritam:ritam /app /app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
USER ritam
ENTRYPOINT ["uvicorn"]
CMD ["ritam.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
