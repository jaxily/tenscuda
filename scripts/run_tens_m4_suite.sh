#!/usr/bin/env bash
# Sequential M4 baseline on card host (one GPU job). Logs under results/tens_m4_suite.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=.venv/bin/python
MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
LOG=results/tens_m4_suite.log
exec > >(tee -a "$LOG") 2>&1
echo "=== tens M4 suite $(date -Is) ==="

run() { echo "--- $* ---"; "$@"; }

run "$PY" scripts/kv_diagnose.py "$MODEL" 8
run "$PY" scripts/trace_cost.py "$MODEL" default
run env TENSCUDA_DECODE_PAUSE_MODES=1 "$PY" scripts/trace_cost.py "$MODEL" default
run "$PY" scripts/m4_replay_probe.py "$MODEL" 32
run env TENSCUDA_DECODE_PAUSE_MODES=1 "$PY" scripts/m4_replay_probe.py "$MODEL" 32
run "$PY" scripts/m4_pause_compare.py
run "$PY" scripts/m4_trace_profile.py "$MODEL" 10
run env TENSCUDA_DECODE_PAUSE_MODES=1 "$PY" scripts/m4_trace_profile.py "$MODEL" 10
run "$PY" scripts/measure_decode_subset.py
run "$PY" scripts/matrix.py "$MODEL" --pause-benchmark
echo "=== M4 suite done $(date -Is) ==="