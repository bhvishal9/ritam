FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN groupadd -r ritam && useradd -r -g ritam ritam --home /app
USER ritam
WORKDIR /app
EXPOSE 8000
COPY --chown=ritam:ritam assets/indexed_chunks.json /app/assets/indexed_chunks.json
COPY --chown=ritam:ritam pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project
COPY --chown=ritam:ritam src /app/src
RUN uv sync --frozen
ENTRYPOINT ["uv", "run", "uvicorn", "ritam.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]