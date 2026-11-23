#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=.venv/bin/python
MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
LOG=results/tens_tune.log
exec > >(tee -a "$LOG") 2>&1
echo "=== tens tune $(date -Is) ==="
run() { echo "--- $* ---"; "$@"; }
run env TENSCUDA_THIN_DECODE=1 "$PY" scripts/m4_pause_parity_gate.py "$MODEL"
run env TENSCUDA_LOADER_PREFETCH=1 "$PY" scripts/loader_bench_prefetch.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
run "$PY" scripts/m4_env_sweep.py
echo "=== tune done $(date -Is) ==="