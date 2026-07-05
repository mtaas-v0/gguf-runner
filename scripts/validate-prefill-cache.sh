#!/usr/bin/env bash
# Prefill-cache oracle: render a blob for a static system prompt, then compare
# greedy output of a cold run vs a cached run. Identical output = correct.
set -u
MODEL="${1:-/Users/jens/tmp/everlock/target/models/Qwen3.5-2B-Q3_K_M.gguf}"
BIN="$(dirname "$0")/../target/release/gguf-runner"
OUT_DIR="$(mktemp -d)"; trap 'rm -rf "$OUT_DIR"' EXIT
rep() { local i out=""; for ((i=0;i<$1;i++)); do out+="$2"; done; printf '%s' "$out"; }
SYS="$(rep 55 'You are a systems engineer for the Everlock server. ')Answer operator questions concisely."
PROMPT="What is 2+2? Reply with just the number."
t() { python3 -c 'import time; print(f"{time.time():.2f}")'; }

t0=$(t); "$BIN" --model "$MODEL" --system-prompt "$SYS" \
  --render-prefill-cache "$OUT_DIR/p.gpfc" >/dev/null 2>&1; t1=$(t)
echo "render: $(python3 -c "print(f'{$t1-$t0:.1f}s')") ($(stat -f%z "$OUT_DIR/p.gpfc" 2>/dev/null || stat -c%s "$OUT_DIR/p.gpfc") bytes)"

t0=$(t); "$BIN" --model "$MODEL" --system-prompt "$SYS" --temperature 0 \
  --max-tokens 12 --prompt "$PROMPT" >"$OUT_DIR/cold" 2>/dev/null; t1=$(t)
echo "cold:   $(python3 -c "print(f'{$t1-$t0:.1f}s')")"

t0=$(t); "$BIN" --model "$MODEL" --system-prompt "$SYS" --temperature 0 \
  --max-tokens 12 --prefill-cache "$OUT_DIR/p.gpfc" --prompt "$PROMPT" \
  >"$OUT_DIR/cached" 2>/dev/null; t1=$(t)
echo "cached: $(python3 -c "print(f'{$t1-$t0:.1f}s')")"

if diff -q "$OUT_DIR/cold" "$OUT_DIR/cached" >/dev/null; then
  echo "verdict: OK (identical greedy output)"; exit 0
else
  echo "verdict: MISMATCH"; diff "$OUT_DIR/cold" "$OUT_DIR/cached"; exit 1
fi
