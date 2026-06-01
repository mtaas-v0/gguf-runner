#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORY_FILE="$ROOT_DIR/docs/tokenizer-benchmark-history.md"
TMP_OUTPUT="$(mktemp)"
SOURCE_DIR="${TOKENIZER_BENCH_SOURCE_DIR:-/Users/jens/tmp/everlock/docs}"
trap 'rm -f "$TMP_OUTPUT"' EXIT

cd "$ROOT_DIR"

{
  cargo test --lib synthetic_benchmark_reports_gpt2_speedup --release -- --ignored --nocapture
  cargo test --lib synthetic_benchmark_reports_sentencepiece_speedup --release -- --ignored --nocapture
  cargo test --lib synthetic_benchmark_reports_gpt2_chunk_1k_speedup --release -- --ignored --nocapture
  cargo test --lib synthetic_benchmark_reports_gpt2_chunk_2k_code_speedup --release -- --ignored --nocapture
  cargo test --lib synthetic_benchmark_reports_sentencepiece_chunk_1k_speedup --release -- --ignored --nocapture
  cargo test --lib synthetic_benchmark_reports_sentencepiece_chunk_2k_code_speedup --release -- --ignored --nocapture
  if [[ -d "$SOURCE_DIR" ]]; then
    TOKENIZER_BENCH_SOURCE_DIR="$SOURCE_DIR" cargo test --lib synthetic_benchmark_reports_gpt2_source_chunk_1200_speedup --release -- --ignored --nocapture
    TOKENIZER_BENCH_SOURCE_DIR="$SOURCE_DIR" cargo test --lib synthetic_benchmark_reports_gpt2_source_chunk_1800_speedup --release -- --ignored --nocapture
    TOKENIZER_BENCH_SOURCE_DIR="$SOURCE_DIR" cargo test --lib synthetic_benchmark_reports_sentencepiece_source_chunk_1200_speedup --release -- --ignored --nocapture
    TOKENIZER_BENCH_SOURCE_DIR="$SOURCE_DIR" cargo test --lib synthetic_benchmark_reports_sentencepiece_source_chunk_1800_speedup --release -- --ignored --nocapture
  fi
} 2>&1 | tee "$TMP_OUTPUT"

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

{
  echo ""
  echo "## $timestamp ($git_rev)"
  echo ""
  echo "| Mode | Docs | Bytes/Doc | Ref Min us | Ref Median us | Ref Max us | Opt Min us | Opt Median us | Opt Max us | Median Speedup x |"
  echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

  grep '^TOKENIZER_BENCH ' "$TMP_OUTPUT" | while read -r line; do
    mode=""
    docs=""
    bytes=""
    ref_min=""
    ref_med=""
    ref_max=""
    opt_min=""
    opt_med=""
    opt_max=""
    speedup=""
    for field in $line; do
      case "$field" in
        mode=*) mode="${field#mode=}" ;;
        docs=*) docs="${field#docs=}" ;;
        bytes_per_doc=*) bytes="${field#bytes_per_doc=}" ;;
        reference_min_us=*) ref_min="${field#reference_min_us=}" ;;
        reference_median_us=*) ref_med="${field#reference_median_us=}" ;;
        reference_max_us=*) ref_max="${field#reference_max_us=}" ;;
        optimized_min_us=*) opt_min="${field#optimized_min_us=}" ;;
        optimized_median_us=*) opt_med="${field#optimized_median_us=}" ;;
        optimized_max_us=*) opt_max="${field#optimized_max_us=}" ;;
        median_speedup_x=*) speedup="${field#median_speedup_x=}" ;;
      esac
    done
    echo "| $mode | $docs | $bytes | $ref_min | $ref_med | $ref_max | $opt_min | $opt_med | $opt_max | $speedup |"
  done
} >> "$HISTORY_FILE"

echo "Appended benchmark results to $HISTORY_FILE"
