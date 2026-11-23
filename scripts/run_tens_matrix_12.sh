#!/usr/bin/env bash
# Tier-1 matrix refresh on card host (sequential). Logs: results/tens_matrix_12.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=.venv/bin/python
LOG=results/tens_matrix_12.log
exec > >(tee -a "$LOG") 2>&1
echo "=== tens matrix 12 $(date -Is) ==="
MODELS=(
  HuggingFaceTB/SmolLM2-135M-Instruct
  HuggingFaceTB/SmolLM2-360M-Instruct
  Qwen/Qwen2.5-0.5B-Instruct
  TinyLlama/TinyLlama-1.1B-Chat-v1.0
  unsloth/Llama-3.2-1B-Instruct
  Qwen/Qwen2.5-1.5B-Instruct
  HuggingFaceTB/SmolLM2-1.7B-Instruct
  unsloth/gemma-2-2b-it
  Qwen/Qwen2.5-3B-Instruct
  unsloth/Llama-3.2-3B-Instruct
  microsoft/Phi-3.5-mini-instruct
  unsloth/Meta-Llama-3.1-8B-Instruct
)
for m in "${MODELS[@]}"; do
  echo "--- matrix $m ---"
  "$PY" scripts/matrix.py "$m" --new-tokens 32 || echo "FAILED: $m" >&2
done
"$PY" scripts/matrix.py --render
echo "=== matrix 12 done $(date -Is) ==="