# Retrieval eval matrix — 2026-06-27

A parameter sweep over the retrieval pipeline, run against the `ducks` eval set
(78 queries, `top_k=3`, generation model `gemini-3.1-flash-lite`, git `5e91b8f`).
Purpose: establish an honest **before-baseline** for a future hybrid-search
milestone, and record real per-query cost.

## What was swept

| Axis | Values |
|---|---|
| Embedding model | `gemini-embedding-001`, `gemini-embedding-2` |
| Chunk size (chars) | 2500, 5000, 10000 |
| Similarity threshold | 0.70–0.85 (both models); **plus 0.55–0.65 for `gemini-embedding-2`** (see calibration note) |

33 cells total. Embedding model and chunk size are **index-time** (each
combination is a separate index build); the threshold is a **query-time**
post-retrieval filter, so a threshold sweep reuses one index.

Total run: **2,574 queries, $0.8439**, zero errored rows after repair.

## Layout

- `matrix_summary.csv` — one row per cell; the analysis surface.
- `<model>_cs-<size>_th-<thr>/` — per cell:
  - `config.json` — params, git SHA, timestamp, per-bucket metrics, cost.
  - `results.json` / `results.csv` — raw per-query eval output.
  - `eval.log` / `index.log` — structured JSON logs (per-stage timings, threshold breaches).

Per-bucket recall (f_easy / f_med / f_hard / multi_hop / oos) is computed here by
grouping on the `id` prefix — the eval harness does not emit it natively.

## The baseline to beat

**`gemini-embedding-001` · chunk 2500 · threshold 0.75** — recall **0.84**,
out-of-scope abstention **0.13**, **$0.00042 / query**. This is the row the
hybrid-search work must improve on.

## Findings

1. **The global cosine threshold trades recall against abstention with no good
   joint point.** Across all 33 cells there is no setting with both recall > 0.8
   and abstention > 0.3. The best compromise (the baseline above) still answers
   ~87% of unanswerable queries with confident, wrong retrievals. Relevant and
   irrelevant chunks share a cosine band (~0.74–0.88 for emb-001), so one global
   cutoff cannot separate them. **This is the architectural gap hybrid search /
   reranking is meant to close.**

2. **A similarity threshold is per-model, not transferable.** `emb-2`'s scores
   sit ~0.06 lower (median top-1 0.74 vs 0.80), so the 0.70–0.85 band tuned for
   `emb-001` lands above `emb-2`'s useful range. Read in its own regime (0.55–0.60),
   `emb-2` reaches recall 0.83 — level with `emb-001`. The naive same-threshold
   comparison is a calibration artifact, not a quality verdict.

3. **`emb-001` + small chunks is the pick.** With each model calibrated, overall
   recall ties (0.84 vs 0.83), but `emb-001` wins multi-hop (0.61 vs 0.39) and
   costs ~3× less ($0.033 vs $0.095 per 78q), because `emb-2`'s competitive recall
   only appears at low threshold + large chunks — more context per generation.
   Chunk preference inverts: `emb-001` peaks small, `emb-2` peaks large.

4. **Cost is driven by retrieved context, not output.** Cost falls monotonically
   as the threshold rises (fewer passing chunks → less generation context); full
   abstention costs $0. Chunk size × top_k is the cost lever.

## Caveats / honesty notes

- The eval score keys only on retrieved doc paths — the generated answer is not
  scored. These are **retrieval** metrics; generation cost is measured but not graded.
- Thresholds are not comparable across embedding models (Finding 2).
- `gemini-embedding-2` was swept at a lower band to be evaluated fairly; the two
  models share only the 0.70–0.85 band.
