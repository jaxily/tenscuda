#!/usr/bin/env bash
# 7B–9B matrix expansion on card host. Uses --compile-bench (TENSCUDA_COMPILE=1)
# so warm tok/s is comparable to the Llama-3.1-8B unlock row.
# Logs: results/tens_matrix_big.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}:$PWD/scripts"
PY=.venv/bin/python
LOG=results/tens_matrix_big.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
echo "=== tens matrix big $(date -Is) ==="
mapfile -t MODELS < <("$PY" -c 'from matrix_models import MATRIX_BIG; print("\n".join(MATRIX_BIG))')
for m in "${MODELS[@]}"; do
  echo "--- matrix --compile-bench $m ---"
  # 7B+ CPU refs are slow/mmap; auto dtype is fine on tens (~30 GiB). Failures continue.
  "$PY" scripts/matrix.py "$m" --new-tokens 32 --compile-bench --timeout 7200 \
    || echo "FAILED: $m" >&2
done
"$PY" scripts/matrix.py --render
echo "=== matrix big done $(date -Is) ==="
