#!/usr/bin/env bash
# After matrix_big + compile_scale: keep measuring more models + decode scale.
# Safe to start while big/scale are running — waits for them first.
# Logs: results/tens_keep_testing.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}:$PWD/scripts"
PY=.venv/bin/python
LOG=results/tens_keep_testing.log
mkdir -p results
exec >>"$LOG" 2>&1
echo "=== keep testing $(date -Is) ==="

wait_idle() {
  local pat=$1
  while pgrep -f "$pat" >/dev/null 2>&1; do
    echo "waiting for $pat … $(date -Is)"
    sleep 60
  done
}

# Wait for earlier suite scripts (not our own keep_testing).
for pat in \
  'scripts/run_tens_matrix_big\.sh' \
  'scripts/run_tens_big_then_scale\.sh' \
  'scripts/run_tens_compile_scale\.sh'
do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "peer suite active: $pat"
    wait_idle "$pat"
  fi
done
wait_idle 'scripts/matrix\.py|scripts/cpu_ref\.py|scripts/load_bench\.py'
echo "card idle at $(date -Is); starting keep-testing suite"

# 1) Fill compile ladder gaps (skips existing compile-bench rows).
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
bash scripts/run_tens_compile_scale.sh || echo "FAILED: compile_scale"

# 2) Extra models beyond MATRIX_BIG (still single-card bf16 candidates).
mapfile -t EXTRA < <("$PY" -c 'from matrix_models import MATRIX_MORE; print("\n".join(MATRIX_MORE))')
for m in "${EXTRA[@]}"; do
  short="${m##*/}"
  json="results/matrix/${short}.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "$json" ]] && \
     "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('tt') and 'compile' in (r.get('notes') or '') else 1)" "$json"; then
    echo "--- skip $m (existing compile-bench) ---"
    continue
  fi
  echo "--- keep-testing compile-bench $m ---"
  "$PY" scripts/matrix.py "$m" --new-tokens 32 --compile-bench --timeout 7200 \
    || echo "FAILED: $m" >&2
done

# 3) Functional decode scale with compile unlock (separate harness numbers).
echo "--- measure_decode --scale --compile ---"
"$PY" scripts/measure_decode_subset.py --scale --compile \
  || echo "FAILED: measure_decode scale"

"$PY" scripts/matrix.py --render
echo "=== keep testing done $(date -Is) ==="

# Final ladder print (COMPILE_SCALE + MATRIX_MORE).
"$PY" - <<'PY'
import json, os
from matrix_models import COMPILE_SCALE, MATRIX_MORE
mdir = "results/matrix"
print("\n=== keep-testing warm tok/s summary ===")
print("model | params | warm tok/s | first run s | notes")
print("---|---|---|---|---")
for mid in list(COMPILE_SCALE) + [m for m in MATRIX_MORE if m not in COMPILE_SCALE]:
    short = mid.split("/")[-1]
    path = os.path.join(mdir, f"{short}.json")
    if not os.path.exists(path):
        print(f"{mid} | - | missing | - | -")
        continue
    rec = json.load(open(path))
    tt = rec.get("tt")
    p = rec.get("params") or 0
    params = f"{p/1e9:.2f}B" if p >= 1e9 else f"{p/1e6:.0f}M"
    if not tt:
        print(f"{mid} | {params} | FAIL | - | {(rec.get('tt_error') or '?')[:80]}")
        continue
    warm = [r["decode_tok_per_s"] for r in tt["runs"][1:]]
    mean = sum(warm) / len(warm)
    first = tt["runs"][0]["total_s"]
    notes = (rec.get("notes") or "").strip() or "ok"
    print(f"{mid} | {params} | {mean:.2f} | {first:.0f} | {notes[:60]}")
PY
