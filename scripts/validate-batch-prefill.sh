#!/usr/bin/env bash
# Validation harness for batched prefill (GGUF_BATCH_PREFILL).
#
# Three runs per prompt at temperature 0 (greedy => deterministic):
#   seq   — GGUF_BATCH_PREFILL=0 (sequential prefill; the reference)
#   exact — batched, GGUF_BATCH_PREFILL_EXACT=1: bitwise mirror of the
#           sequential kernels. MUST equal seq — any difference is a
#           structural bug in the batching. This is the pass/fail gate.
#   fast  — batched, default kernels: K-quant matmuls use the
#           dequantize-once kernel (~1e-6 summation reorder). Output may
#           rarely flip on near-ties; reported informationally with timing.
#
# Usage:
#   scripts/validate-batch-prefill.sh [path/to/model.gguf]
#   SKIP_LONG=1 scripts/validate-batch-prefill.sh    # skip the ~2K-token prompt
#
# Exit code: 0 when all EXACT runs match seq, 1 otherwise.

set -u

MODEL="${1:-/Users/jens/tmp/everlock/target/models/Qwen3.5-2B-Q3_K_M.gguf}"
BIN="$(dirname "$0")/../target/release/gguf-runner"
MAX_TOKENS=12
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not built (run: cargo build --release --bin gguf-runner)" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL" >&2
  exit 2
fi

rep() { # rep <count> <text>
  local i out=""
  for ((i = 0; i < $1; i++)); do out+="$2"; done
  printf '%s' "$out"
}

SHORT_PROMPT="What is 2+2? Reply with just the number."
MED_PROMPT="$(rep 60 'You are a systems engineer. ')Now answer: what is 2+2? Reply with just the number."
LONG_PROMPT="$(rep 280 'You are a systems engineer. ')Now answer: what is 2+2? Reply with just the number."

NAMES=(short medium)
PROMPTS=("$SHORT_PROMPT" "$MED_PROMPT")
if [[ -z "${SKIP_LONG:-}" ]]; then
  NAMES+=(long)
  PROMPTS+=("$LONG_PROMPT")
fi

run_one() { # run_one <batch> <exact> <outfile> <prompt> -> prints elapsed seconds
  local t0 t1
  t0=$(python3 -c 'import time; print(f"{time.time():.2f}")')
  GGUF_BATCH_PREFILL="$1" GGUF_BATCH_PREFILL_EXACT="$2" "$BIN" \
    --model "$MODEL" --temperature 0 --max-tokens "$MAX_TOKENS" \
    --prompt "$4" >"$3" 2>/dev/null
  t1=$(python3 -c 'import time; print(f"{time.time():.2f}")')
  python3 -c "print(f'{$t1 - $t0:.1f}')"
}

fail=0
printf '%-8s %9s %9s %9s   %-14s %s\n' prompt "seq(s)" "exact(s)" "fast(s)" "exact-gate" "fast-info"
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  prompt="${PROMPTS[$i]}"
  t_seq=$(run_one 0 0 "$OUT_DIR/$name.seq" "$prompt")
  t_exa=$(run_one 1 1 "$OUT_DIR/$name.exa" "$prompt")
  t_fas=$(run_one 1 0 "$OUT_DIR/$name.fas" "$prompt")
  if diff -q "$OUT_DIR/$name.seq" "$OUT_DIR/$name.exa" >/dev/null; then
    gate="OK (bitwise)"
  else
    gate="STRUCT-BUG"
    fail=1
  fi
  if diff -q "$OUT_DIR/$name.seq" "$OUT_DIR/$name.fas" >/dev/null; then
    info="identical"
  else
    info="differs (rounding)"
  fi
  printf '%-8s %9s %9s %9s   %-14s %s\n' "$name" "$t_seq" "$t_exa" "$t_fas" "$gate" "$info"
  if [[ "$gate" == STRUCT-BUG ]]; then
    echo "--- sequential ---"; cat "$OUT_DIR/$name.seq"
    echo "--- exact batched ---"; cat "$OUT_DIR/$name.exa"
  fi
done
exit "$fail"
