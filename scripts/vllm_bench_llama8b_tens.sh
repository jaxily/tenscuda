#!/usr/bin/env bash
# Concurrent batch TPS for Llama-3.1-8B via vLLM-TT (metal env). One card only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/card_hygiene.sh
source "$REPO_ROOT/scripts/lib/card_hygiene.sh"

MODEL="${VLLM_MODEL:-${HF_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}}"
MESH="${MESH_DEVICE:-N150}"
TT_EXTRA="${VLLM_TT_EXTRA_CONFIG:-{\"tt\": {\"trace_region_size\": 67108864}}}"
LOG="${VLLM_BENCH_LOG:-/tmp/vllm_bench_8b.log}"
SERVE_LOG="${VLLM_SERVE_LOG:-/tmp/vllm_serve_8b.log}"
PORT="${VLLM_PORT:-8000}"

TT_METAL="${TT_METAL_HOME:-$HOME/tt-metal}"
PLUGIN="${VLLM_TT_PLUGIN_DIR:-$HOME/vllm-tt-plugin}"

if [[ ! -d "$TT_METAL" ]]; then
  echo "Missing TT_METAL at $TT_METAL" >&2
  exit 1
fi
if [[ ! -d "$PLUGIN" ]]; then
  echo "Missing vllm-tt-plugin at $PLUGIN" >&2
  exit 1
fi

echo "=== Card hygiene (stop TensCUDA / stale vLLM) ==="
card_hygiene || true

# shellcheck disable=SC1091
source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL" PYTHONPATH="$TT_METAL" MESH_DEVICE="$MESH" HF_MODEL="$MODEL"
cd "$PLUGIN"

: >"$SERVE_LOG"
: >"$LOG"

cleanup() {
  if [[ -n "${SRV:-}" ]]; then
    kill "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting server (MESH_DEVICE=$MESH, max_num_seqs=32) …"
nohup python examples/server_example_tt.py \
  --model "$MODEL" \
  --max_num_seqs 32 \
  --additional-config "$TT_EXTRA" \
  >"$SERVE_LOG" 2>&1 &
SRV=$!

if ! wait_vllm_health "$PORT" 180 2; then
  echo "ERROR: vLLM health check failed on :$PORT" >&2
  tail -50 "$SERVE_LOG" || true
  exit 1
fi

if vllm_serve_log_has_engine_dead "$SERVE_LOG"; then
  echo "ERROR: EngineDeadError during server startup — free the card and retry vLLM alone" >&2
  echo "  bash scripts/run_tt_speed_showcase.sh --skip-metal" >&2
  exit 1
fi

echo "Running vllm bench serve → $LOG"
set +e
vllm bench serve \
  --backend openai \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 32 \
  --ignore-eos \
  2>&1 | tee "$LOG"
bench_rc=${PIPESTATUS[0]}
set -e

if grep -qE 'EngineDeadError|engine dead' "$SERVE_LOG" "$LOG" 2>/dev/null; then
  echo "WARN: EngineDeadError detected — parse may be empty; rerun after metal pytest cools down" >&2
fi

grep -Ei 'throughput|tok/s|tps|request' "$LOG" | tail -20 || true
exit "$bench_rc"