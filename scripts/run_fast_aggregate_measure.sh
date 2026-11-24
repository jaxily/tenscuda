#!/usr/bin/env bash
# Aggregate serving proof on one P150: metal b32 + vLLM offline + vLLM serve bench.
# Writes showcase + per-path JSON under results/. Card host only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${HF_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
SKIP_SERVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-serve) SKIP_SERVE=1; shift ;;
    -h|--help)
      echo "Usage: run_fast_aggregate_measure.sh [--skip-serve]"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

export HF_MODEL="$MODEL"
SHOWCASE_ARGS=(--serve)
if [[ "$SKIP_SERVE" -eq 1 ]]; then
  SHOWCASE_ARGS=()
fi

echo "=== A) metal + vLLM offline (+ serve via run_tt_speed_showcase) ==="
bash "$REPO_ROOT/scripts/run_tt_speed_showcase.sh" "${SHOWCASE_ARGS[@]}"

OFFLINE_LOG="${SHOWCASE_OFFLINE_LOG:-/tmp/vllm_offline_b32_speed.log}"
SERVE_LOG="${VLLM_BENCH_LOG:-/tmp/vllm_bench_8b.log}"
PY="${TT_METAL_HOME:-$HOME/tt-metal}/python_env/bin/python"

if [[ -f "$OFFLINE_LOG" ]] && grep -qE 'tok/s|toks/s' "$OFFLINE_LOG" 2>/dev/null; then
  echo "=== B) export vLLM offline JSON ==="
  "$PY" "$REPO_ROOT/scripts/export_vllm_offline_json.py" "$OFFLINE_LOG"
fi

if [[ "$SKIP_SERVE" -eq 0 ]] && [[ -f "$SERVE_LOG" ]]; then
  echo "=== C) export vLLM serve JSON ==="
  "$PY" "$REPO_ROOT/scripts/export_vllm_serve_json.py" "$SERVE_LOG" "$MODEL" || true
fi

echo "=== D) refresh docs/TEST_RESULTS.md ==="
"$PY" "$REPO_ROOT/scripts/render_test_results.py"
echo "Done — see results/showcase_*.json and docs/TEST_RESULTS.md"