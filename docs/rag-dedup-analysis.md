# RAG Chunk Dedup Analysis

This repository includes a small real-source analysis flow for the exact chunk
deduplication stage in RAG index building.

## What gets measured

- chunk counts from `chunk_directory(...)` at `max_chars=1200` and `1800`
- unique chunk counts after applying the same exact-text dedup logic used by
  `RagIndex::build_from_dir`
- duplicate chunk count and duplicate ratio
- number of repeated chunk texts and the maximum reuse count for any single
  chunk body

The analysis output is emitted by ignored tests in `src/rag/mod.rs` as
structured `RAG_DEDUP ...` lines.

## How to run

From repo root:

```sh
docs/run-rag-dedup-analysis.sh
```

That script:

- runs the source-derived dedup analysis in `--release`
- uses `RAG_DEDUP_SOURCE_DIR` when set, otherwise defaults to
  `/Users/jens/tmp/everlock/docs`
- appends a timestamped markdown table to `docs/rag-dedup-history.md`

## Why this exists

The new dedup stage only helps if the chunker actually emits repeated chunk
texts. This flow keeps a simple in-repo record of real corpus duplication so we
can judge whether the optimization is material on a given docs tree.

## Interpreting results

- `Duplicate Ratio` is the share of chunk instances avoided by exact-text dedup.
- `Reused Texts` counts how many distinct chunk bodies appear more than once.
- `Max Reuse` shows the hottest repeated chunk body.

Low duplication means the dedup stage is harmless but unlikely to move total
index time much. High duplication means repeated tokenizer and embedding work is
being removed.
