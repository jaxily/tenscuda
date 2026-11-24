#!/usr/bin/env bash
# Start vLLM-TT OpenAI-compatible server on tt-metal env (32 concurrent seqs).
# Optional: run vllm bench serve against it. One card only — see docs/CUDA_TO_METAL.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/card_hygiene.sh
source "$REPO_ROOT/scripts/lib/card_hygiene.sh"

MODEL="${VLLM_MODEL:-${HF_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}}"
MESH="${MESH_DEVICE:-N150}"
PORT="${VLLM_PORT:-8000}"
TT_EXTRA="${VLLM_TT_EXTRA_CONFIG:-{\"tt\": {\"trace_region_size\": 67108864}}}"
SERVE_LOG="${VLLM_SERVE_LOG:-/tmp/vllm_serve_8b.log}"
BENCH_LOG="${VLLM_BENCH_LOG:-/tmp/vllm_bench_8b.log}"
RUN_BENCH=0
FOREGROUND=0

usage() {
  cat <<EOF
Usage: run_tt_serve_fast.sh [OPTIONS]

  (default)     card hygiene + background server on 127.0.0.1:$PORT
  --bench       also run scripts/vllm_bench_llama8b_tens.sh (starts server internally)
  --foreground  run server in foreground (no bench; Ctrl-C to stop)

Env: TT_METAL_HOME, VLLM_TT_PLUGIN_DIR, HF_MODEL, MESH_DEVICE, VLLM_PORT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench) RUN_BENCH=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

TT_METAL="${TT_METAL_HOME:-$HOME/tt-metal}"
PLUGIN="${VLLM_TT_PLUGIN_DIR:-$HOME/vllm-tt-plugin}"

if [[ ! -d "$TT_METAL" ]]; then
  echo "Missing TT_METAL at $TT_METAL — see docs/TT_METAL_FASTPATH.md" >&2
  exit 1
fi
if [[ ! -d "$PLUGIN" ]]; then
  echo "Missing vllm-tt-plugin at $PLUGIN — run scripts/setup_vllm_tt_tens.sh" >&2
  exit 1
fi

echo "=== Card hygiene ==="
card_hygiene || true

if [[ "$RUN_BENCH" -eq 1 ]]; then
  exec bash "$REPO_ROOT/scripts/vllm_bench_llama8b_tens.sh"
fi

# shellcheck disable=SC1091
source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL" PYTHONPATH="$TT_METAL" MESH_DEVICE="$MESH" HF_MODEL="$MODEL"
cd "$PLUGIN"

: >"$SERVE_LOG"
echo "Starting vLLM-TT server (model=$MODEL, max_num_seqs=32, port=$PORT) …"
if [[ "$FOREGROUND" -eq 1 ]]; then
  exec python examples/server_example_tt.py \
    --model "$MODEL" \
    --max_num_seqs 32 \
    --additional-config "$TT_EXTRA" \
    2>&1 | tee "$SERVE_LOG"
fi

nohup python examples/server_example_tt.py \
  --model "$MODEL" \
  --max_num_seqs 32 \
  --additional-config "$TT_EXTRA" \
  >"$SERVE_LOG" 2>&1 &
SRV=$!
echo "PID $SRV — log $SERVE_LOG"

if ! wait_vllm_health "$PORT" 180 2; then
  echo "ERROR: server did not become healthy on :$PORT" >&2
  tail -40 "$SERVE_LOG" 2>/dev/null || true
  kill "$SRV" 2>/dev/null || true
  exit 1
fi

if vllm_serve_log_has_engine_dead "$SERVE_LOG"; then
  echo "ERROR: EngineDeadError in serve log — run card_hygiene and retry without metal pytest" >&2
  kill "$SRV" 2>/dev/null || true
  exit 1
fi

echo "OK: http://127.0.0.1:$PORT/health"
echo "Bench: bash scripts/vllm_bench_llama8b_tens.sh  # or export and run vllm bench serve"
echo "Stop:  kill $SRV"