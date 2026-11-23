#!/usr/bin/env bash
# One-card Tenstorrent speed showcase: tt-metal batch-32 + vLLM-TT offline (+ optional serve).
# Produces logs under /tmp and JSON under results/ via parse_tt_speed_logs.py.
#
# Prereq on card: ~/tt-metal built, ~/vllm-tt-plugin installed (see TT_METAL_FASTPATH.md).
# Do NOT run TensCUDA matrix concurrently (CHIP_IN_USE).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${HF_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MESH="${MESH_DEVICE:-N150}"
RUN_SERVE=0
SKIP_METAL=0
SKIP_OFFLINE=0
QUICK=0

usage() {
  cat <<'EOF'
Usage: run_tt_speed_showcase.sh [OPTIONS]

  (default)     metal batch-32 pytest + vLLM offline inference
  --serve       also run vllm bench serve (longer; needs vLLM server)
  --quick       skip metal pytest; parse existing /tmp logs only
  --skip-metal
  --skip-offline

Env: TT_METAL_HOME, HF_MODEL, MESH_DEVICE, SHOWCASE_CPU_REF_TOK_S (default 3.05)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve) RUN_SERVE=1; shift ;;
    --quick) QUICK=1; shift ;;
    --skip-metal) SKIP_METAL=1; shift ;;
    --skip-offline) SKIP_OFFLINE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

METAL_LOG="${SHOWCASE_METAL_LOG:-/tmp/tt_metal_b32_speed.log}"
OFFLINE_LOG="${SHOWCASE_OFFLINE_LOG:-/tmp/vllm_offline_b32_speed.log}"
SERVE_LOG="${VLLM_BENCH_LOG:-/tmp/vllm_bench_8b.log}"

echo "=== Card hygiene (stop TensCUDA / duplicate pytest) ==="
pkill -f 'scripts/matrix.py' 2>/dev/null || true
pkill -f 'run_tens_' 2>/dev/null || true
sleep 1
if pgrep -af 'pjrt-plugin|tenscuda.*matrix' >/dev/null 2>&1; then
  echo "WARN: TensCUDA/matrix may still be running — check with: pgrep -af matrix" >&2
fi

TT_METAL="${TT_METAL_HOME:-$HOME/tt-metal}"
if [[ ! -d "$TT_METAL" ]]; then
  echo "Missing TT_METAL at $TT_METAL" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL" PYTHONPATH="$TT_METAL" HF_MODEL="$MODEL" MESH_DEVICE="$MESH"

if [[ "$QUICK" -eq 0 && "$SKIP_METAL" -eq 0 ]]; then
  echo "=== A) tt-metal batch-32 (pytest -k; brackets break ::node ids in some shells) ==="
  cd "$TT_METAL"
  : >"$METAL_LOG"
  if ! pytest models/ttt_compat/demo/simple_text_demo.py \
    -k "performance and batch-32 and not log-probs" -s 2>&1 | tee "$METAL_LOG"; then
    echo "WARN: metal pytest failed — will still parse log / prior /tmp metrics if present" >&2
  fi
fi

if [[ "$QUICK" -eq 0 && "$SKIP_OFFLINE" -eq 0 ]]; then
  PLUGIN="${VLLM_TT_PLUGIN_DIR:-$HOME/vllm-tt-plugin}"
  if [[ ! -d "$PLUGIN" ]]; then
    echo "Missing vllm-tt-plugin at $PLUGIN — run scripts/setup_vllm_tt_tens.sh" >&2
    exit 1
  fi
  echo "=== B) vLLM-TT offline (32 concurrent, 128 new tokens) ==="
  cd "$PLUGIN"
  export HF_MODEL MESH_DEVICE
  : >"$OFFLINE_LOG"
  if ! python examples/offline_inference_tt.py \
    --model "$MODEL" \
    --prompts_json examples/prompts.json \
    --num_repeat_prompts 8 \
    --max_seqs_in_batch 32 \
    --greedy_sampling \
    --max_tokens 128 \
    --max_model_len 8192 \
    --additional-config '{"tt": {"trace_region_size": 67108864}}' \
    2>&1 | tee "$OFFLINE_LOG"; then
    echo "WARN: vLLM offline failed — parse step may use prior /tmp log if non-empty" >&2
  fi
fi

if [[ "$RUN_SERVE" -eq 1 ]]; then
  echo "=== C) vLLM bench serve ==="
  bash "$REPO_ROOT/scripts/vllm_bench_llama8b_tens.sh" 2>&1 | tee "$SERVE_LOG" || true
fi

PARSE_ARGS=(--model "$MODEL")
[[ "$SKIP_METAL" -eq 1 ]] && PARSE_ARGS+=(--skip-metal)
[[ "$SKIP_OFFLINE" -eq 1 ]] && PARSE_ARGS+=(--skip-offline)
[[ "$RUN_SERVE" -eq 1 ]] && PARSE_ARGS+=(--include-serve)
PARSE_ARGS+=(--metal-log "$METAL_LOG" --vllm-offline-log "$OFFLINE_LOG" --vllm-serve-log "$SERVE_LOG")
# If offline run died mid-log, fall back to newest committed vLLM artifact in repo.
if ! grep -qE '[0-9]+\.[0-9]+\s*tok/s' "$OFFLINE_LOG" 2>/dev/null; then
  FALLBACK="$(ls -t "$REPO_ROOT"/results/vllm_offline_b32_speed_*.json 2>/dev/null | head -1 || true)"
  if [[ -n "${FALLBACK:-}" ]]; then
  echo "WARN: using vLLM metrics from $FALLBACK" >&2
    PARSE_ARGS+=(--vllm-json "$FALLBACK")
  fi
fi

echo "=== Parse → results/showcase_*.json ==="
OUT="$("$TT_METAL/python_env/bin/python" "$REPO_ROOT/scripts/parse_tt_speed_logs.py" "${PARSE_ARGS[@]}")"
echo "Wrote $OUT"