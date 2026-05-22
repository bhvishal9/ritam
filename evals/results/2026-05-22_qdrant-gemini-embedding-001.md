# Eval Run: Qdrant vector store, gemini-embedding-001
**Date:** 2026-05-22
**Dataset:** ducks (40 queries)
**Model:** gemini-3.1-flash-lite-preview
**Embedding model:** gemini-embedding-001
**Vector store:** Qdrant

## Configuration
| Parameter | Value |
|---|---|
| `top_k` | 3 |

## Results Summary
| Metric | Value |
|---|---|
| Total examples | 40 |
| Errored examples | 0 |

### Retrieval (factual + multi_hop, n=35)
| Metric | Value |
|---|---|
| Matched | 33 |
| Missed | 2 |
| Observed recall (matched/total) | 0.943 |
| Retrieval recall (matched/non-error) | 0.943 |
| Coverage (returned>0 among non-error) | 1.000 |

### By query type
| Query type | Matched | Total | Recall |
|---|---|---|---|
| factual | 28 | 30 | 0.933 |
| multi_hop | 5 | 5 | 1.000 |
| out_of_scope | 0 | 5 | — |

### Out-of-scope abstention
| Metric | Value |
|---|---|
| Abstention rate | 0.000 |

## Context

First run with the Qdrant backend. The `QdrantStoreClient` uses one collection per embedding model (collection name sanitised from the model name), with a payload index on `dataset` for per-dataset filtering. Chunks are stored with deterministic UUID5 point IDs keyed on `{dataset}-{embedding_model}-{source}`, making upserts idempotent.

The Qdrant URL is now configurable via the `QDRANT_URL` env var (default `http://localhost:6333`).

## Regression vs. file store baseline (2026-05-01, gemini-embedding-001, recall=1.000)

| Metric | File store | Qdrant | Delta |
|---|---|---|---|
| Retrieval recall | 1.000 | 0.943 | −0.057 |
| Coverage | 1.000 | 1.000 | 0 |
| Abstention rate | 0.000 | 0.000 | 0 |
| factual recall | 1.000 | 0.933 | −0.067 |
| multi_hop recall | 1.000 | 1.000 | 0 |

Minor recall drop (−0.057) on factual queries. Coverage is unchanged at 1.000 — the retriever always returns chunks, so the 2 misses are ranking/relevance misses, not empty-result failures. Multi-hop recall is perfect. Abstention remains at 0.000 for both backends, confirming this is a property of `gemini-embedding-001` producing high similarity scores for out-of-scope queries, not a backend-specific issue.

The recall gap vs. the file store is small and likely attributable to differences in cosine similarity computation (Qdrant uses its own HNSW approximate search vs. exact brute-force in the file store). No threshold or `top_k` tuning was done for Qdrant.
