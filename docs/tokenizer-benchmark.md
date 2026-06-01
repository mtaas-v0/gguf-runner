# Tokenizer Benchmarking

This repository includes synthetic tokenizer benchmarks that compare the current
tokenizer path against a legacy reference implementation with the older merge
loop behavior.

## What gets measured

- GPT-2-style BPE tokenization on a fixed synthetic markdown corpus
- SentencePiece-style tokenization on a fixed synthetic markdown corpus
- GPT-2-style tokenization on 1 KB and 2 KB corpus-shaped markdown chunks
- SentencePiece-style tokenization on 1 KB and 2 KB corpus-shaped markdown chunks
- Optional source-derived markdown chunk cases at 1200 B and 1800 B from
  `TOKENIZER_BENCH_SOURCE_DIR`
- 1 warmup run + 7 measured runs per case
- Min, median, and max wall-clock duration per implementation

The benchmark output is emitted by ignored tests in
`src/engine/tokenizer/mod.rs` as structured `TOKENIZER_BENCH ...` lines.

## How to run

From repo root:

```sh
docs/run-tokenizer-bench.sh
```

That script:

- runs the synthetic tokenizer benchmarks in `--release`
- uses `TOKENIZER_BENCH_SOURCE_DIR` when set, otherwise defaults to
  `/Users/jens/tmp/everlock/docs` if present
- captures the structured benchmark lines
- appends a timestamped markdown table to
  `docs/tokenizer-benchmark-history.md`

## Why this exists

Tokenizer micro-optimizations are noisy when judged from single debug-mode test
runs. This flow keeps a simple in-repo record so changes can be compared over
time with the same benchmark shape and a stable output format.

## Interpreting results

- `Median Speedup x > 1.0` means the current tokenizer path is faster than the
  legacy reference on the benchmark's median run.
- `Median Speedup x < 1.0` means the current path is slower on that benchmark.
- Min/max spread is useful for spotting noisy runs and unstable measurements.

These synthetic cases are meant for relative regression tracking, not absolute
production throughput claims.

The `*_chunk_1k` and `*_chunk_2k_code` cases are intended to better resemble
real RAG indexing inputs:

- `chunk_1k`: wiki/runbook-style prose and list sections around 1 KB
- `chunk_2k_code`: larger mixed markdown plus fenced config/code blocks around 2 KB
- `source_chunk_1200` / `source_chunk_1800`: paragraph-packed chunks from a
  real markdown directory, approximating common `--rag-max-chars-per-chunk`
  values
