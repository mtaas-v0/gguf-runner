#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORY_FILE="$ROOT_DIR/docs/rag-dedup-history.md"
TMP_OUTPUT="$(mktemp)"
SOURCE_DIR="${RAG_DEDUP_SOURCE_DIR:-/Users/jens/tmp/everlock/docs}"
trap 'rm -f "$TMP_OUTPUT"' EXIT

cd "$ROOT_DIR"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

{
  RAG_DEDUP_SOURCE_DIR="$SOURCE_DIR" cargo test --lib report_source_chunk_dedup_1200 --release -- --ignored --nocapture
  RAG_DEDUP_SOURCE_DIR="$SOURCE_DIR" cargo test --lib report_source_chunk_dedup_1800 --release -- --ignored --nocapture
} 2>&1 | tee "$TMP_OUTPUT"

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

{
  echo ""
  echo "## $timestamp ($git_rev)"
  echo ""
  echo "| Source Dir | Source Files | Max Chars | Chunks | Unique Chunks | Duplicate Chunks | Duplicate Ratio | Reused Texts | Max Reuse |"
  echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

  grep '^RAG_DEDUP ' "$TMP_OUTPUT" | while read -r line; do
    source_dir=""
    source_files=""
    max_chars=""
    chunks=""
    unique_chunks=""
    duplicate_chunks=""
    duplicate_ratio=""
    reused_texts=""
    max_reuse=""
    for field in $line; do
      case "$field" in
        source_dir=*) source_dir="${field#source_dir=}" ;;
        source_files=*) source_files="${field#source_files=}" ;;
        max_chars=*) max_chars="${field#max_chars=}" ;;
        chunks=*) chunks="${field#chunks=}" ;;
        unique_chunks=*) unique_chunks="${field#unique_chunks=}" ;;
        duplicate_chunks=*) duplicate_chunks="${field#duplicate_chunks=}" ;;
        duplicate_ratio=*) duplicate_ratio="${field#duplicate_ratio=}" ;;
        reused_texts=*) reused_texts="${field#reused_texts=}" ;;
        max_reuse=*) max_reuse="${field#max_reuse=}" ;;
      esac
    done
    echo "| $source_dir | $source_files | $max_chars | $chunks | $unique_chunks | $duplicate_chunks | $duplicate_ratio | $reused_texts | $max_reuse |"
  done
} >> "$HISTORY_FILE"

echo "Appended dedup results to $HISTORY_FILE"
