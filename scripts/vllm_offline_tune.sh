#!/usr/bin/env bash
# Quick vLLM-TT offline sweeps on one P150 (aggregate tok/s). One job on card.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/card_hygiene.sh
source "$REPO_ROOT/scripts/lib/card_hygiene.sh"

MODEL="${HF_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MESH="${MESH_DEVICE:-N150}"
PLUGIN="${VLLM_TT_PLUGIN_DIR:-$HOME/vllm-tt-plugin}"
TT_METAL="${TT_METAL_HOME:-$HOME/tt-metal}"
LOG_DIR="${VLLM_TUNE_LOG_DIR:-/tmp/vllm_tune}"
TS="$(date -u +%Y-%m-%dT%H%M)"

mkdir -p "$LOG_DIR"
card_hygiene || true

# shellcheck disable=SC1091
source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL" PYTHONPATH="$TT_METAL" MESH_DEVICE="$MESH" HF_MODEL="$MODEL"
cd "$PLUGIN"

run_one() {
  local tag="$1" max_seqs="$2" trace_mb="$3"
  local trace_bytes=$((trace_mb * 1024 * 1024))
  local log="$LOG_DIR/vllm_tune_${tag}_${TS}.log"
  echo "=== $tag (max_seqs=$max_seqs trace=${trace_mb}MiB) ==="
  python examples/offline_inference_tt.py \
    --model "$MODEL" \
    --prompts_json examples/prompts.json \
    --num_repeat_prompts 8 \
    --max_seqs_in_batch "$max_seqs" \
    --greedy_sampling \
    --max_tokens 128 \
    --max_model_len 8192 \
    --additional-config "{\"tt\": {\"trace_region_size\": ${trace_bytes}}}" \
    2>&1 | tee "$log"
  PYTHONPATH="$REPO_ROOT/scripts" "$TT_METAL/python_env/bin/python" - <<PY
from pathlib import Path
from parse_tt_speed_logs import parse_vllm_offline
import json, socket
from datetime import datetime, timezone
text = Path("$log").read_text(errors="replace")
row = parse_vllm_offline(text)
row.update({"tag": "$tag", "max_seqs_in_batch": $max_seqs, "trace_region_mb": $trace_mb, "log": "$log"})
out = Path("$REPO_ROOT/results/vllm_tune_${tag}_${TS}.json")
out.write_text(json.dumps(row, indent=2) + "\n")
print(out, row.get("tqdm_output_tok_s_final"), row.get("engine_peak_generation_tok_s"))
PY
}

# Baseline + slight trace bump (P150 prefill wants >= ~61MB; 64 default)
run_one "b32_t64" 32 64
# Heavier concurrency if engine allows
run_one "b32_t128" 32 128 || true

echo "Logs: $LOG_DIR; JSON: $REPO_ROOT/results/vllm_tune_*_${TS}.json"