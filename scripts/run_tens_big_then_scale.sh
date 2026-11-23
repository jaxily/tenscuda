#!/usr/bin/env bash
# Wait for run_tens_matrix_big.sh (if running), then fill the compile tok/s ladder.
# Skips models already measured with --compile-bench. Logs: results/tens_big_then_scale.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
LOG=results/tens_big_then_scale.log
exec >>"$LOG" 2>&1
echo "=== big-then-scale waiter $(date -Is) ==="

wait_idle() {
  local pat=$1
  while pgrep -f "$pat" >/dev/null 2>&1; do
    echo "waiting for $pat … $(date -Is)"
    sleep 60
  done
}

# If matrix_big is still going, wait it out. If not started, that is fine — scale still runs.
if pgrep -f 'scripts/run_tens_matrix_big\.sh' >/dev/null 2>&1; then
  echo "matrix_big in progress; waiting…"
  wait_idle 'scripts/run_tens_matrix_big\.sh'
  # also wait for any leftover matrix/cpu_ref/load_bench from the last model
  wait_idle 'scripts/matrix\.py|scripts/cpu_ref\.py|scripts/load_bench\.py'
  echo "matrix_big idle at $(date -Is)"
else
  echo "matrix_big not running; starting compile scale now"
fi

export SKIP_EXISTING="${SKIP_EXISTING:-1}"
bash scripts/run_tens_compile_scale.sh
echo "=== big-then-scale done $(date -Is) ==="
