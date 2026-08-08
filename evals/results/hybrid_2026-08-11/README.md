# Hybrid search + reranking — first eval runs, 2026-08-11

First evals of the hybrid-retrieval milestone: dense + BM25 sparse vectors fused
with Reciprocal Rank Fusion, followed by a cross-encoder rerank. The global cosine
`similarity_threshold` is replaced by a `reranking_threshold` applied to
cross-encoder scores.

Run against the `ducks` 78-query set, `top_k=3`, generation model
`gemini-3.1-flash-lite`, embedding model `gemini-embedding-001`, working tree on
top of git `5557303` (uncommitted).

## Pipeline under test

| Stage | Before (baseline) | This milestone |
|---|---|---|
| Retrieval | dense only (cosine) | dense + BM25 sparse, RRF fusion |
| Ranking | cosine score | `jinaai/jina-reranker-v1-turbo-en` cross-encoder |
| Filtering | `similarity_threshold` (cosine, 0–1) | `reranking_threshold` (logits, unbounded) |

**The two thresholds are not comparable.** Cosine similarities and cross-encoder
logits are different scales; only rank-free metrics (recall, abstention) can be
compared across the two regimes.

## Runs

All four cells have full `config.json` / `results.json` / `results.csv` preserved.
The three cs-1500 cells were re-run to recover raw output after an earlier set was
overwritten; every metric reproduced its original value exactly, which is a useful
reproducibility check in its own right.

| Chunk size | Threshold | Recall@3 | factual | multi-hop | Abstention | Coverage | $/query | Directory |
|---|---|---|---|---|---|---|---|---|
| 1500 | −0.5 | 0.730 | 0.956 | 0.167 | **0.400** | 0.968 | $0.000204 | `cs-1500_th-0.5/` |
| 1500 | −1.0 | 0.810 | 0.978 | 0.389 | 0.133 | 0.984 | $0.000247 | `cs-1500_th-1.0/` |
| 1500 | −2.0 | 0.841 | **1.000** | 0.444 | 0.000 | 1.000 | $0.000304 | `cs-1500_th-2.0/` |
| 2500 | −0.5 | 0.571 | 0.756 | 0.111 | 0.467 | 0.810 | $0.000223 | `cs-2500_th-0.5/` |
| *baseline: 2500, cosine 0.75* | | *0.841* | *0.933* | *0.611* | *0.133* | *1.000* | *$0.000420* | `../matrix_2026-06-27/` |

Index for the cs-1500 runs: 52 points (vs 30 at cs-2500) over the same 7 documents.

## Findings

### 1. Chunk size 1500 beats 2500 decisively, and the mechanism is truncation

At a fixed threshold of −0.5, moving from 1500 to 2500 characters costs 9 factual
queries (0.956 → 0.756) and drops coverage from 0.968 to 0.810.

The cause is the reranker's context window. `jina-reranker-v1-turbo-en` advertises
8K context, but fastembed pins its tokenizer truncation to **512 tokens** (~2,000
characters) — verified directly:

```
tokenizer truncation: {'max_length': 512, ...}

answer at END of  1500 chars -> score -2.178
answer at END of  2500 chars -> score -3.323
answer at END of  5000 chars -> score -4.068
answer at END of 10000 chars -> score -4.068   <- identical: stopped reading
```

A 2500-character chunk gets clipped, so any chunk whose relevant passage sits past
the cut scores ~1.1 lower. With a cutoff at −0.5 that pushes correct answers below
the line. The failure breakdown confirms it: **10 of 11 factual misses returned
zero chunks**, not wrong ones. Retrieval found the candidates; the reranker scored
them under the threshold.

Supporting: 38 of the 96 surviving scores sit inside [−0.5, 0.0) — 40% of what
survives is within half a point of the cutoff. At 2500 the whole result balances
on the threshold.

### 2. Multi-hop is broken by ranking, not by filtering

Multi-hop recall is bad at every setting tried: 0.111–0.444 against a baseline of
0.611. Two independent pieces of evidence show the threshold is not the cause.

At cs-1500 / −2.0, **coverage is 1.000 and abstention is 0.000** — nothing is
being filtered at all — and multi-hop still only reaches 0.444. That is the
ceiling of the current pipeline.

And the failure modes differ by query type in the same run:

```
factual  : 11 misses -> 10 returned NOTHING,  1 returned wrong docs
multi_hop: 16 misses ->  2 returned nothing, 14 returned WRONG docs
```

Every multi-hop question needs exactly 2 distinct documents. Measured on the
cs-1500 / −0.5 run: 11 of 18 queries returned chunks from **one** document, and
only 3 of 18 hit both required documents. Typical shape:

```
need = [duck_wars_extended_history.md, isoprene_planetary_survey.md]
 got = [duck_wars_extended_history.md, duck_wars_extended_history.md,
        duck_wars_extended_history.md]
```

A cross-encoder scores each chunk independently against the query and has no view
of the result set. Whichever document matches the query text best produces several
strong chunks and takes every slot. Dense-only retrieval diversified partly by
accident — chunk embeddings from different documents occupy different regions —
and reranking removes that accident. The lever is diversity-aware selection
(per-document cap, MMR, or Qdrant `query_points_groups`), not threshold tuning.

### 3. The threshold traces a frontier that crosses the baseline curve

At cs-1500, lowering the threshold trades abstention for recall monotonically.
Compared against the baseline curve:

- **−2.0** — recall 0.841 (equal to baseline) at abstention 0.000 (baseline
  0.133). Dominated.
- **−1.0** — abstention 0.133 (equal to baseline) at recall 0.810 (baseline
  0.841). Dominated.
- **−0.5** — recall 0.730 at abstention 0.400. The baseline matrix jumps from
  abstention 0.133 / recall 0.841 straight to abstention 0.667 / recall 0.429;
  interpolating puts the old curve near 0.635 recall at abstention 0.40. **Not
  dominated.**

So the two curves cross: the hybrid pipeline wins in the high-abstention region
and loses in the high-recall region. The 2026-06-27 matrix concluded that no
cosine threshold reached recall > 0.8 with abstention > 0.3; factual recall of
0.956 at abstention 0.400 is the first configuration to open that region, and it
is what reranking was added to buy.

### 4. Cost per query is down across the whole frontier

Generation cost scales with how much chunk text reaches the prompt, so halving
chunk size cuts input tokens even when the same number of chunks is returned:

| Config | Recall | Abstention | $/query |
|---|---|---|---|
| baseline (cs-2500, cosine 0.75) | 0.841 | 0.133 | $0.000420 |
| cs-1500, −2.0 | 0.841 | 0.000 | **$0.000304** (−28%) |
| cs-1500, −1.0 | 0.810 | 0.133 | **$0.000247** (−41%) |

At **identical recall** (0.841), the hybrid pipeline at cs-1500 costs 28% less per
query. At **identical abstention** (0.133) it costs 41% less but gives up 3.8
points of recall.

Two caveats. The saving comes from chunk size, not from hybrid retrieval or
reranking — a dense-only cs-1500 run would likely show the same reduction, which
is one more reason the ablation matters. And this counts *generation* cost only:
reranking adds roughly 160 ms of CPU per query, which is a real cost on a
per-CPU-second serverless platform and is not captured in `$/query`.

## What these runs do NOT establish

Three variables changed at once relative to the baseline — chunk size, hybrid
retrieval, and reranking. **The multi-hop regression cannot yet be attributed**
between smaller chunks and the reranker's concentration behaviour, and neither can
the abstention gain.

The outstanding ablation is **dense-only at cs-1500**, which isolates the reranker.
No claim of the form "hybrid search improved X" is defensible until it has run.

## Next

1. Record pre-threshold candidate scores in the eval output, so a threshold sweep
   becomes an offline recompute over one run rather than N full runs.
2. Run the dense-only cs-1500 ablation.
3. Fix multi-hop with diversity-aware selection; re-sweep the threshold after.
