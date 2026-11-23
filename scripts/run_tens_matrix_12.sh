#!/usr/bin/env bash
# Tier-1 matrix refresh on card host (sequential). Logs: results/tens_matrix_12.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}:$PWD/scripts"
PY=.venv/bin/python
LOG=results/tens_matrix_12.log
exec > >(tee -a "$LOG") 2>&1
echo "=== tens matrix 12 $(date -Is) ==="
mapfile -t MODELS < <("$PY" -c 'from matrix_models import MATRIX_12; print("\n".join(MATRIX_12))')
for m in "${MODELS[@]}"; do
  echo "--- matrix $m ---"
  "$PY" scripts/matrix.py "$m" --new-tokens 32 || echo "FAILED: $m" >&2
done
"$PY" scripts/matrix.py --render
echo "=== matrix 12 done $(date -Is) ==="
