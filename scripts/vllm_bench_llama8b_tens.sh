#!/usr/bin/env bash
# Concurrent batch TPS for Llama-3.1-8B via vLLM-TT (metal env). One card only.
set -euo pipefail

MODEL="${VLLM_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MESH="${MESH_DEVICE:-N150}"
# P150 prefill trace needs >50MB trace region (vLLM default); 64MB worked in a smoke run.
TT_EXTRA='{"tt": {"trace_region_size": 67108864}}'
LOG="${VLLM_BENCH_LOG:-/tmp/vllm_bench_8b.log}"
PORT="${VLLM_PORT:-8000}"

TT_METAL="${TT_METAL_HOME:-$HOME/tt-metal}"
# shellcheck disable=SC1091
source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL" PYTHONPATH="$TT_METAL"
cd ~/vllm-tt-plugin

echo "Starting server (MESH_DEVICE=$MESH) …"
nohup python examples/server_example_tt.py \
  --model "$MODEL" \
  --max_num_seqs 32 \
  > /tmp/vllm_serve_8b.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 5
done

echo "Running vllm bench serve → $LOG"
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

grep -Ei 'throughput|tok/s|tps|request' "$LOG" | tail -20 || true