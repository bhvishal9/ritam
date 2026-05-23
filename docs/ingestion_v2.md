# Ingestion v2 — Incremental Ingestion

## Overview

`IngestionService` implements incremental ingestion: on each run it compares the current state of the source directory against what was previously indexed (tracked in SQLite), and only re-indexes documents that are new, updated, or deleted.

## Change detection

Each document is assigned an `index_fingerprint` — a SHA-256 hash of:

```
{chunk_size}-{chunk_separator}-{doc_content}
```

This means any of the following triggers a re-index of the affected document:
- File content changes
- `chunk_size` changes
- `chunk_separator` changes

A change to `embedding_model` triggers a full re-index of all documents because the SQLite records are keyed on `(doc_path, dataset, embedding_model)` — the new model has no prior records.

## Known limitation: non-atomic writes

`process_docs()` writes to two separate stores — Qdrant and SQLite — in sequence:

```
1. Build chunks (embed via LLM)
2. Delete old vectors from Qdrant
3. Store new vectors in Qdrant
4. Update SQLite (commit)
```

This is **not atomic**. If a failure occurs after step 3 but before step 4, Qdrant and SQLite will be out of sync: vectors exist in Qdrant but SQLite still shows the old fingerprint, causing the document to appear as "updated" on the next run and triggering a redundant re-index.

**Mitigation:** Re-running `process_docs()` is safe — the re-index will overwrite the stale vectors and the SQLite record will be corrected on the next successful commit.

A proper fix would require a two-phase commit or a saga pattern across both stores. Deferred to a later phase.
