#!/usr/bin/env bash
# Compile-unlock tok/s ladder: small → 9B. Answers whether warm decode rises with size.
# Each row: matrix.py --compile-bench. Logs: results/tens_compile_scale.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}:$PWD/scripts"
PY=.venv/bin/python
LOG=results/tens_compile_scale.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
echo "=== tens compile scale $(date -Is) ==="
# SKIP_EXISTING=1 (default): reuse rows that already have compile-bench TT results
# (e.g. after run_tens_matrix_big.sh), so the ladder only fills gaps.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
mapfile -t MODELS < <("$PY" -c 'from matrix_models import COMPILE_SCALE; print("\n".join(COMPILE_SCALE))')
for m in "${MODELS[@]}"; do
  short="${m##*/}"
  json="results/matrix/${short}.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$json" ]] && \
     "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('tt') and 'compile' in (r.get('notes') or '') else 1)" "$json"; then
    echo "--- skip $m (existing compile-bench row) ---"
    continue
  fi
  echo "--- compile-bench $m ---"
  "$PY" scripts/matrix.py "$m" --new-tokens 32 --compile-bench --timeout 7200 \
    || echo "FAILED: $m" >&2
done
"$PY" scripts/matrix.py --render
echo "=== compile scale done $(date -Is) ==="
# Quick summary table from results/matrix JSON (warm tok/s only).
"$PY" - <<'PY'
import json, os
from matrix_models import COMPILE_SCALE
mdir = "results/matrix"
print("\nmodel | params | warm tok/s (compile) | first run s | status")
print("---|---|---|---|---")
for mid in COMPILE_SCALE:
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
        print(f"{mid} | {params} | FAIL | - | {(rec.get('tt_error') or '?')[:60]}")
        continue
    warm = [r["decode_tok_per_s"] for r in tt["runs"][1:]]
    mean = sum(warm) / len(warm)
    first = tt["runs"][0]["total_s"]
    notes = rec.get("notes") or ""
    status = "compile" if "compile" in notes else "ok"
    print(f"{mid} | {params} | {mean:.2f} | {first:.0f} | {status}")
PY
