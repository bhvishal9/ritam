# Ablation: dense-only at chunk size 1500

The controlled comparison the hybrid milestone was missing. Same corpus, same
78-query `ducks` set, same embedding model, same `top_k=3` and `candidate_k=9`,
same **chunk size 1500** — the only difference from `hybrid_2026-08-11/` is the
retrieval pipeline.

Run from a detached checkout of the pre-hybrid commit `a969592`, with
`naive_rag.py` chunk-size default set to 1500. Index rebuilt from scratch:
52 points, single unnamed dense vector, no sparse config.

| | this ablation | `hybrid_2026-08-11/` | `matrix_2026-06-27/` (emb-001) |
|---|---|---|---|
| Retrieval | dense only (cosine) | dense + BM25 sparse, RRF | dense only (cosine) |
| Ranking | cosine score | cross-encoder rerank | cosine score |
| Chunk size | **1500** | **1500** | 2500 |

Comparing this against `hybrid_2026-08-11` isolates **hybrid + reranking**.
Comparing this against `matrix_2026-06-27` isolates **chunk size**.

## Results

| threshold | recall@3 | factual | multi-hop | abstention | coverage | $/query |
|---|---|---|---|---|---|---|
| 0.70 | 0.841 | 1.000 | 0.444 | 0.000 | 1.000 | $0.000312 |
| 0.75 | 0.841 | 1.000 | 0.444 | 0.000 | 1.000 | $0.000311 |
| 0.78 | 0.825 | 0.978 | 0.444 | 0.200 | 0.984 | $0.000258 |
| 0.80 | 0.698 | 0.867 | 0.278 | 0.600 | 0.889 | $0.000179 |
| 0.85 | 0.095 | 0.133 | 0.000 | 0.933 | 0.143 | $0.000021 |

## Finding 1 — hybrid search and reranking added nothing

At the unfiltered ceiling (coverage 1.000, abstention 0.000) the two pipelines are
**identical on every metric**:

| | recall | factual | multi-hop | coverage |
|---|---|---|---|---|
| dense-only, cosine 0.70 | 0.841 | 1.000 | 0.444 | 1.000 |
| hybrid + rerank, logit −2.0 | 0.841 | 1.000 | 0.444 | 1.000 |

Not close — the same numbers. With filtering off, both pipelines put the same
documents in the top 3 on all 63 retrieval queries.

Across the abstention frontier, dense-only is **slightly ahead** everywhere
(dense values interpolated between measured points):

| abstention | dense-only recall | hybrid recall |
|---|---|---|
| 0.000 | 0.841 | 0.841 |
| 0.133 | ~0.831 | 0.810 |
| 0.400 | ~0.762 | 0.730 |

So hybrid retrieval plus a cross-encoder reranker is not merely neutral on this
corpus — it is marginally worse at equal abstention, while adding ~160 ms of
CPU-bound reranking to every query and a 150 MB model to the deployment.

**This is a negative result and it should be reported as one.**

## Finding 2 — the earlier conclusions were wrong, and chunk size explains everything

Two claims recorded in `hybrid_2026-08-11/README.md` do not survive this ablation:

- *"Reranking opened the abstention > 0.3 region"* — **false.** Dense-only at
  cs-1500 reaches abstention 0.200 at recall 0.825 and abstention 0.600 at recall
  0.698. Compare the cs-2500 dense-only curve, which fell from recall 0.841 at
  abstention 0.133 straight to recall 0.429 at abstention 0.667. **Halving the
  chunk size opened that region, not reranking.**
- *"Reranking caused the multi-hop regression"* — **false.** Dense-only at cs-1500
  scores multi-hop 0.444, the same as hybrid. The regression from the 0.611
  baseline is entirely a chunk-size effect.

The real trade-off is chunk size, and it cuts both ways:

| | factual | multi-hop | overall |
|---|---|---|---|
| cs-2500 (dense) | 0.933 | **0.611** | 0.841 |
| cs-1500 (dense) | **1.000** | 0.444 | 0.841 |

Smaller chunks make every factual query succeed and cost 6 of 18 multi-hop
queries. Overall recall is identical at 0.841 — the two effects cancel exactly,
which is why the aggregate number hid this until the buckets were split out.

Mechanism: smaller chunks mean more chunks per document, so a `top_k=3` with no
per-document diversity constraint is more likely to fill all three slots from one
document. That fails multi-hop, which always needs two distinct documents here.
The `2026-05-31_threshold-0.70.md` write-up flagged this same concentration
problem; halving the chunk size made it worse.

## Finding 3 — cost is a chunk-size effect too

Dense-only cs-1500 costs $0.000312/query at recall 0.841, against $0.000420 for
the cs-2500 baseline at the same recall — a **26% reduction with no pipeline
change at all**. The hybrid pipeline's $0.000304 at the same recall is within
noise of the dense-only number. The saving was never attributable to hybrid
search.

## Caveats — why this is a result about *this corpus*

The corpus is 7 documents / 52 chunks, and `candidate_k=9` is roughly 17% of the
entire index. A reranker earns its keep by reordering a candidate pool much larger
than what is returned; here the pool already contains nearly everything relevant,
so there is little for it to fix. Likewise BM25 has almost no inverse-document-
frequency signal to work with across 7 documents.

So the honest claim is **"hybrid search and reranking did not pay for themselves
on this corpus at this scale"**, not "hybrid search does not work". Testing the
hypothesis properly needs a corpus where dense retrieval genuinely misses — more
documents, rarer exact-match terms, a candidate pool that is a small fraction of
the index.

At cs-1500 the reranker was *not* truncating (1500 chars ≈ 375 tokens, inside the
512-token window), so this result is not contaminated by the truncation issue that
degraded the cs-2500 hybrid run.

## What to do with this

1. **Do not ship hybrid + reranking on this evidence.** It costs latency, memory
   and cold-start time and returns nothing measurable.
2. The multi-hop / factual trade-off is the live problem, and it is a `top_k`
   diversity problem, not a retrieval-strategy problem. Per-document caps, MMR, or
   Qdrant `query_points_groups` address it directly and cost nothing at inference.
3. If hybrid search is worth keeping in the portfolio, it needs a corpus that can
   demonstrate the failure it is designed to fix.
