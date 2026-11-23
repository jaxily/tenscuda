#!/usr/bin/env bash
# Same-shape vLLM bench as scripts/vllm_bench_llama8b_tens.sh, for NVIDIA CUDA GPUs.
# Writes /tmp/nvidia_vllm_bench_ref.log and results/nvidia_ref_<ts>.json when run from repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${VLLM_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
PORT="${VLLM_PORT:-8000}"
LOG="${NVIDIA_BENCH_LOG:-/tmp/nvidia_vllm_bench_ref.log}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — run on a CUDA host or set PATH." >&2
  exit 1
fi

echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -z "${SKIP_SERVER:-}" ]]; then
  echo "Starting vLLM OpenAI server on CUDA (port $PORT) …"
  nohup vllm serve "$MODEL" --port "$PORT" --max-num-seqs 32 > /tmp/nvidia_vllm_serve.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 2
  done
fi

: >"$LOG"
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

TS="$(date -u +%Y-%m-%dT%H%MZ)"
OUT="$REPO_ROOT/results/nvidia_ref_${TS}.json"
mkdir -p "$REPO_ROOT/results"
python3 "$REPO_ROOT/scripts/parse_tt_speed_logs.py" \
  --skip-metal \
  --skip-offline \
  --include-serve \
  --vllm-serve-log "$LOG" \
  --model "$MODEL" \
  --out "$OUT"

echo "Wrote $OUT"