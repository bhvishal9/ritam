# Evals

This directory contains the retrieval eval set and a CLI to run it against the RAG service.

The eval set is the contract that says whether a change to ingestion, embedding, or retrieval made things better or worse. Treat it as part of the code, not a scratch file.

## Run

From the repository root:

```bash
uv run python evals/run_eval.py
```

Prerequisites: the dataset must already be indexed, and the environment must carry
`LLM_API_KEY`, `VECTOR_STORE`, and `QDRANT_URL` (plus `QDRANT_API_KEY` for a non-loopback URL).
The runner queries whatever collection `LLM_EMBEDDING_MODEL_NAME` resolves to — run the index
command with the same embedding model first, or every row errors.

Options:

- `--top-k INTEGER` — default retrieval depth when a row does not set `top_k` (default: `3`)
- `--dataset-file PATH` — dataset file to load (default: `dataset.jsonl`, resolved relative to `evals/`)

## What the eval set is designed to measure

This is a fictional, in-universe corpus by deliberate choice. Real-world topics let the LLM fall back on pretraining recall, which makes retrieval quality unmeasurable. With fictional content, every correct answer is forced through the retrieval pipeline, so observed scores actually reflect retrieval behavior.

The set is built on four design principles:

1. **Stratification.** Three query types (`factual`, `multi_hop`, `out_of_scope`) test three different failure modes. Within `factual`, three difficulty tiers (`f_easy_*`, `f_med_*`, `f_hard_*`) embedded in the `id` let you tell whether a regression is hitting easy keyword matches or only the hard cases.
2. **Per-bucket sample size.** Each bucket is large enough that a single wrong answer does not dominate the score. The `out_of_scope` bucket is the smallest, with 15 rows — enough that a single false retrieval moves the bucket by ~7 points rather than ~17.
3. **Hard negatives, not just easy positives.** The hard factual rows include lexical traps (many "Squeak"-named entities; "bubble" appears in three unrelated contexts), precise numeric facts buried deep in long documents, and queries that paraphrase rather than quote the source. These are where retrieval pipelines actually fail in production.
4. **Adversarial out-of-scope.** The unanswerable queries are not random nonsense. They are things the corpus explicitly marks as unknown (Precursor morphology, Wisp memory mechanism), things just outside what the corpus states (Hydromarch salary, Suds's post-armistice fate), and plausible-sounding queries about non-entities (Vice-Hydromarch). Those are the queries that tempt a retriever into a confident wrong answer.

## Dataset format

JSONL, one object per line.

Required fields:

- `id` (string) — see ID convention below.
- `dataset` (string) — currently always `"ducks"`.
- `query` (string) — the user-facing query.
- `expected_docs` (list of strings) — zero or more doc paths, **relative to the dataset
  directory** (`duck_technology.md`, not `assets/ducks/duck_technology.md`). Matching is exact
  string comparison against what the retriever returns, so a prefixed path silently never
  matches and looks like a retrieval failure. Empty for `out_of_scope`.
- `query_type` (string) — one of `factual`, `multi_hop`, `out_of_scope`.

Optional field:

- `top_k` (integer, `>= 1`) — per-row override for the CLI `--top-k`.

The script uses strict Pydantic validation (`extra="forbid"`); unknown fields will fail to load.

### Match semantics per `query_type`

| `query_type` | Matches when |
|---|---|
| `factual` | Any path in `expected_docs` appears in `returned_docs` |
| `multi_hop` | All paths in `expected_docs` appear in `returned_docs` |
| `out_of_scope` | Zero docs returned |

### `id` convention

The `id` is the only place difficulty is encoded. The script does not parse it, so post-hoc stratified reporting is done by grepping the results file by prefix.

| Prefix | Meaning |
|---|---|
| `f_easy_*` | Easy factual: keyword overlap with the source, single-doc answer |
| `f_med_*` | Medium factual: specific fact buried in a longer doc, may require light inference |
| `f_hard_*` | Hard factual: lexical traps, disambiguation across similarly named entities, precise numeric or named-entity recall |
| `m_*` | Multi-hop: answer genuinely requires combining facts from two docs |
| `o_*` | Out-of-scope: corpus does not contain a defensible answer |

Example rows:

```json
{"id":"f_easy_bubble_shield","dataset":"ducks","query":"What is the Bubble Shield and how does it dissipate enemy fire?","expected_docs":["duck_technology.md"],"query_type":"factual","top_k":3}
{"id":"m_waddles_birth_and_role","dataset":"ducks","query":"Where was Commodore Waddles born, and what was his decisive contribution to the Battle of the Loofah Line?","expected_docs":["isoprene_planetary_survey.md","duck_wars_extended_history.md"],"query_type":"multi_hop","top_k":3}
{"id":"o_hydromarch_salary","dataset":"ducks","query":"What is the annual salary of the Hydromarch in Drops?","expected_docs":[],"query_type":"out_of_scope","top_k":3}
```

## Current distribution

| Bucket | Rows |
|---|---|
| `f_easy_*` | 20 |
| `f_med_*` | 15 |
| `f_hard_*` | 10 |
| `m_*` | 18 |
| `o_*` | 15 |
| **Total** | **78** |

All seven docs in `assets/ducks/` are covered, weighted toward the larger and more cross-referenced docs (`isoprene_planetary_survey.md` and `duck_wars_extended_history.md`).

## Corpus

| File | Size | Notes |
|---|---|---|
| `the_duck_wars.md` | small | original short war summary |
| `duck_technology.md` | small | original short tech summary |
| `species_compendium.md` | small | seven sapient species, cross-referenced from elsewhere |
| `quack_federation_governance.md` | medium | government, economy, named officials |
| `galactic_gazetteer.md` | medium | sector-by-sector planetary list |
| `duck_wars_extended_history.md` | large | 12-chapter war history with dates, named officers, casualty figures |
| `isoprene_planetary_survey.md` | huge | 9-part reference work on the capital world, deep hierarchy |

Multi-hop questions are designed around cross-document facts. For example, Commodore Waddles's birthplace (Stillwater-Mouth) is in the planetary survey; his war role is in the extended history; both must be retrieved to answer `m_waddles_birth_and_role`.

## Outputs

Each run overwrites:

- `evals/results.json`
- `evals/results.csv`

Per row, output includes:

- `matched` — boolean, per the type-specific match semantics above
- `returned_docs` — list of returned doc paths
- `returned_scores` — cross-encoder score per returned chunk, rounded to 4 dp
- `num_returned` — count of returned docs
- `error` — `null` on success, or an error label (e.g. `rate_limit`)
- `input_cost_usd` / `output_cost_usd` / `total_cost_usd` — generation cost for the row, or `null` when the run couldn't be priced
- `model` — generation model that answered the row

Cost is per-row, so `results.json` supports a cost-per-query figure alongside the quality metrics — that's the point of carrying it here rather than only in the logs. Note it covers generation only; embedding cost is not tracked.

## Summary metrics

The CLI prints, in this order:

- `Total examples` and `Errored examples`
- A retrieval block over **factual + multi_hop rows only**:
  - `Matched` / `Missed`
  - `Observed recall` = `matched / retrieval total` — errors count against you
  - `Retrieval recall` = `matched / non-error retrieval rows`
  - `Coverage` = fraction of non-error retrieval rows where `num_returned > 0`
- `Abstention rate` over `out_of_scope` rows = fraction where the system correctly returned nothing
- A per-`query_type` breakdown (factual / multi_hop / out_of_scope), when more than one type is present
- `Error breakdown` by error class

The two recall figures differ only when rows errored, and the gap between them is the signal: a healthy run has them equal. Observed recall is the honest number to quote, because it refuses to hide rate-limited rows.

What the script still does *not* emit is the difficulty tiers — `f_easy` / `f_med` / `f_hard` are encoded only in the `id`, and the breakdown groups by `query_type`, so all three collapse into one `factual` number. For tier-level reporting, post-process `results.json` by grouping on the `id` prefix. A factual score that holds steady while the hard tier quietly collapses is exactly the failure this dataset is built to expose, and the aggregate will not show it to you.

## Interpreting changes

A few rules of thumb when comparing a run against a baseline:

- A drop in `f_easy_*` recall is a real problem — these queries should always succeed.
- A drop in `f_hard_*` while `f_easy_*` holds usually points at the embedding model, chunking strategy, or top-k.
- A drop in `m_*` while `f_*` holds usually points at chunk overlap or top-k being too low.
- A drop in `o_*` (i.e. the retriever started returning docs for unanswerable queries) is a calibration regression — usually a `RERANKING_THRESHOLD` change. That threshold is now an unbounded cross-encoder logit, not the old 0–1 cosine similarity; the two scales are not comparable, so a value carried over from a pre-hybrid config will behave nothing like its old self.
- An across-the-board drop with high error rate is an infrastructure or rate-limit issue, not a retrieval issue.

## Output streams and exit codes

- Per-row error logs are written to stderr; summary and progress logs to stdout.
- Exit codes: `0` success, `1` dataset/IO/validation error, `2` LLM auth error, `3` other LLM error.

## Baseline and regression checks

`baseline.json` is the reference run that `check_regression.py` compares new results against. Regenerate it only when the dataset changes, or after a deliberate improvement you want to anchor at — a baseline refreshed to make a red gate go green is worse than no gate.

```bash
uv run python evals/check_regression.py
```

It gates on three metrics — `retrieval_recall`, `coverage`, `abstention_rate` — and fails on *any* decrease, with no tolerance band. That strictness is intentional at this corpus size: with 78 rows a single flipped query is a real movement, not noise.

The gate runs in CI on every push and PR to `main` (`.github/workflows/regression-gate.yml`) against an ephemeral Qdrant service container, rebuilding the index from the in-repo corpus each run. That's what makes it hermetic — the gate measures the code under test rather than the drifting state of a long-lived cloud collection. Results are uploaded as a build artifact even when the gate fails.

Note that the gate reads `results.json`, so `run_eval.py` must have run first; it compares whatever is on disk and has no way to tell a fresh run from a stale one.
