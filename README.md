# ritam

A RAG service built to make the cost of an LLM query visible. Every request reports what it
spent — tokens, per-stage latency, dollars — and every retrieval change has to justify itself
against an eval set before it lands.

The RAG part is deliberately unremarkable. The interesting part is the surrounding measurement:
a stratified eval set, a regression gate in CI, per-request cost accounting, and a set of
write-ups recording what actually moved the numbers (including the changes that didn't).

## Blog post (walkthrough)

I wrote up the v1 architecture, trade-offs, and what I learned here:

- https://vbhargava.org/writing/llm-lab-rag-v1/

The repo has moved on since that post — hybrid retrieval, incremental ingestion, and cost
accounting all landed afterwards.

## How a query flows

```
query → embed (Gemini) → Qdrant: dense + BM25 sparse, fused with RRF
      → cross-encoder rerank → threshold filter → top_k → prompt → Gemini → answer + cost
```

`candidate_k` is `top_k × 3`, capped at 10. Anything scoring below `RERANKING_THRESHOLD` is
dropped, and if nothing survives the service abstains rather than answering from an empty
context — abstention is a measured metric, not an accident.

Indexing is a separate, local path. `IngestionService` fingerprints each document
(`sha256(chunk_size + separator + content)`), keeps the state in a SQLite ledger keyed on
`(doc_path, dataset, embedding_model)`, and only re-embeds what changed. Re-running an index is
idempotent and cheap; changing the embedding model forces a full rebuild, by design. Details in
[docs/ingestion_v2.md](docs/ingestion_v2.md).

## Architecture

Layers, enforced by `import-linter` — imports only ever flow downward:

```
api → core → document_source → retrieval → vector_store → llm → cost → (observability | config)
```

The contract lives in `pyproject.toml` and runs in CI and pre-commit, so a violation fails the
build rather than getting caught in review. `core` holds the orchestration (`RagService`,
`IngestionService`), `vector_store` and `llm` are behind protocols, and factories in
`core/factories.py` are the only place that reads settings to pick an implementation.

Qdrant is the vector store. The file-based store is still in the tree behind
`VECTOR_STORE=file` but it's deprecated — it exists for the older eval runs, not for use.

## Getting started

### Prerequisites

- Python 3.14+
- `uv`
- A Google API key for Gemini
- Qdrant — local (`docker run -p 6333:6333 qdrant/qdrant`) or Qdrant Cloud

### Install

```bash
git clone https://github.com/bhvishal9/ritam.git
cd ritam
uv sync
```

### Configure

Settings are loaded from the environment or a `.env` file, and validated as one object at
startup — a bad config fails immediately with the offending field named, rather than halfway
through a request.

| Variable | Default | Notes |
|---|---|---|
| `LLM_API_KEY` | *(required)* | Gemini API key |
| `SOURCE_URI` | — | `file://` URI of the docs root; required for indexing |
| `VECTOR_STORE` | `qdrant` | `qdrant` or `file` (deprecated) |
| `QDRANT_URL` | — | Required when `VECTOR_STORE=qdrant` |
| `QDRANT_API_KEY` | — | Required only for a non-loopback Qdrant URL |
| `LLM_MODEL_NAME` | `gemini-3.1-flash-lite` | Generation model |
| `LLM_EMBEDDING_MODEL_NAME` | `gemini-embedding-001` | Also names the Qdrant collection |
| `RERANKING_THRESHOLD` | `-0.5` | Cross-encoder logit, unbounded — not a cosine score |
| `RERANKING_MODEL` | `jinaai/jina-reranker-v1-turbo-en` | fastembed cross-encoder |

One trap worth knowing: the threshold used to be a cosine similarity in 0–1 and is now a
cross-encoder logit. The two scales aren't comparable, so a threshold copied from an older
config will behave nothing like you expect.

### Index a dataset

`SOURCE_URI` points at the docs root; the dataset name is the subdirectory under it. With
`SOURCE_URI=file:///path/to/assets`, this indexes `assets/ducks/**/*.md`:

```bash
uv run python -m ritam.naive_rag index --dataset ducks
```

| Option | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Dataset to index; also the source subdirectory |
| `--chunk-size` | `1500` | Chunk size in characters |
| `--chunk-separator` | `\n\n` | String used to split chunks |

It prints the diff it applied — new, updated, unchanged, deleted, chunks embedded.

### Query from the CLI

```bash
uv run python -m ritam.naive_rag query --dataset ducks
```

### Run the API

```bash
uv run uvicorn ritam.main:app --reload
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `POST /echo` | Round-trip check |
| `POST /query` | Answer a question against an indexed dataset |

```json
{
  "query": "What is the Bubble Shield?",
  "dataset": "ducks",
  "embedding_model": "gemini-embedding-001",
  "top_k": 3
}
```

`embedding_model` is required and selects the collection — a dataset indexed with one model is
not readable with another. `top_k` must be 1–10. The response carries the answer plus the
source chunks used, and every response carries an `x-request-id` header that ties back to the
logs.

Errors are mapped from a typed taxonomy at the client edge, not by string-matching provider
messages:

| Condition | Status |
|---|---|
| Dataset or collection not indexed | 404 |
| Vector store failure | 500 |
| LLM rate limit | 429 |
| LLM auth / invalid request / unavailable | 502 |

## Observability

Logs are single-line JSON. Per-request fields live in one mutable dict behind a single
`ContextVar` — a detail that matters, because FastAPI runs a plain `def` endpoint on a worker
thread with a *copy* of the context, and per-field context vars written inside the handler would
be invisible to the middleware logging afterwards. Both sides point at the same dict, so the
summary log sees everything the handler recorded.

Each query carries `request_id`, `dataset`, `top_k`, `candidate_k`, `chunks_returned`, stage
timings (`embed_ms`, `retrieve_ms`, `generate_ms`), `model`, token counts, and
`input_cost_usd` / `output_cost_usd` / `total_cost_usd`.

Cost is `Cost | None`, never a `Cost` with null fields. If the model has no pricing entry or the
provider didn't report token counts, the result is `None` and the cost fields stay absent — an
unpriced query must not be silently counted as a free one. Embedding cost isn't tracked; it's
negligible next to generation, and the Gemini Developer API doesn't populate usage on that path
anyway.

## Evals

`evals/` holds a 78-query set over a deliberately fictional corpus. Fictional is the whole
point: with real-world topics the model answers from pretraining and retrieval quality becomes
unmeasurable. Queries are stratified into factual (easy/medium/hard), multi-hop, and
out-of-scope buckets, with adversarial negatives — lexical traps, plausible questions about
entities that don't exist — because an aggregate score hides exactly the regressions worth
catching.

```bash
uv run python evals/run_eval.py
uv run python evals/check_regression.py
```

The gate compares `results.json` against the committed `baseline.json` on retrieval recall,
coverage, and abstention, and fails on any drop. It runs in CI against an ephemeral Qdrant
container so the index is rebuilt from the in-repo corpus each time — the gate measures the code
under test, not the state of a long-lived cloud collection.

Full methodology, dataset format, and match semantics: [evals/README.md](evals/README.md). Every
run is written up under `evals/results/`, superseded conclusions included.

## Deployment

Terraform in `infra/` builds an Artifact Registry repo, a dedicated service account, Secret
Manager entries for `llm_api_key` / `qdrant_url` / `qdrant_api_key`, and a Cloud Run service that
reads them via `value_source`.

```bash
cd infra
terraform init
terraform apply -var="app_version=<image-tag>"
```

Only the query path is deployed. Indexing runs locally against cloud Qdrant — the ledger is
local too, and there's no scheduled or event-driven ingestion in the cloud. That's a deliberate
scope call, not an oversight.

The service has no `allUsers` invoker binding: it's a real LLM endpoint with a real bill behind
it, so calls need an identity token.

The image is a two-stage build — the builder carries a compiler toolchain for any dependency
without a 3.14 wheel, and the runtime is a slim image with no compiler and no `uv`, running as a
non-root user straight out of the baked venv.

## Development

```bash
uv run pytest              # with coverage
uv run ruff check .
uv run ruff format .
uv run mypy src/           # strict
uv run lint-imports        # layer contract
```

All of it runs in pre-commit and again in CI. These are the floor, not the goal — passing them
says the code is well-formed, not that it's good.

## Known gaps

- Per-bucket metrics aren't emitted by the eval runner yet; stratified reporting means
  post-processing `results.json` by `id` prefix.
- `top_k` has no per-document diversity constraint, so all three slots can come from one
  document — which is why multi-hop recall lags factual. Per-document caps or MMR is next.
- The file-based vector store is deprecated but still present.
- No caching layer, so repeated identical queries pay full generation cost every time.
