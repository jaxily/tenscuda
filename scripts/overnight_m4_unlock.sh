#!/usr/bin/env bash
# Overnight M4 unlock runner on the card host.
# Waits for matrix/cpu_ref/load_bench to finish, then runs compile + speed probes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=.venv/bin/python
LOG=results/overnight_m4_unlock.log
mkdir -p results
exec >>"$LOG" 2>&1

echo "=== overnight M4 unlock start $(date -Is) ==="

echo "waiting for matrix / cpu_ref / load_bench to finish..."
while pgrep -f 'scripts/run_tens_matrix_12\.sh|scripts/run_tens_matrix_big\.sh|scripts/run_tens_compile_scale\.sh|scripts/run_tens_big_then_scale\.sh|scripts/run_tens_keep_testing\.sh|scripts/matrix\.py|scripts/cpu_ref\.py|scripts/load_bench\.py' >/dev/null 2>&1; do
  sleep 30
done
sleep 5
echo "card idle at $(date -Is)"

run() {
  echo "--- $* ---"
  "$@" || echo "FAILED: $*"
}

run "$PY" scripts/m4_compile_staticcache_probe.py HuggingFaceTB/SmolLM2-135M-Instruct 16
run "$PY" scripts/trace_cost.py HuggingFaceTB/SmolLM2-135M-Instruct default
run "$PY" scripts/m4_replay_probe.py HuggingFaceTB/SmolLM2-135M-Instruct 32
run env TENSCUDA_DECODE_PAUSE_MODES=1 "$PY" scripts/m4_replay_probe.py HuggingFaceTB/SmolLM2-135M-Instruct 32
run env TENSCUDA_THIN_DECODE=1 "$PY" scripts/m4_pause_parity_gate.py HuggingFaceTB/SmolLM2-135M-Instruct

# If any compile variant matched baseline, try a longer decode timing on the best label noted in JSON.
if [[ -f results/m4_compile_staticcache_SmolLM2-135M-Instruct.json ]]; then
  echo "--- E5 result summary ---"
  "$PY" - <<'PY'
import json, os
from pathlib import Path
p = Path("results/m4_compile_staticcache_SmolLM2-135M-Instruct.json")
d = json.loads(p.read_text())
print("baseline", d.get("baseline_tokens"))
winners = []
for v in d.get("variants", []):
    brief = {k: v.get(k) for k in ("label", "parity_vs_baseline", "tokens_match", "mean_decode_ms", "wall_s", "error")}
    print(brief)
    if v.get("tokens_match") is True and v.get("label") != "baseline_static_greedy":
        winners.append(v)
Path("results/m4_e5_winners.json").write_text(json.dumps(winners, indent=2) + "\n")
print("winners:", [w.get("label") for w in winners])
PY

  if [[ -s results/m4_e5_winners.json ]] && [[ "$(cat results/m4_e5_winners.json)" != "[]" ]]; then
    echo "--- E5 winners: follow-up 32-token generate with matching env ---"
    # Prefer fastest matching mean_decode_ms
    BEST=$("$PY" - <<'PY'
import json
from pathlib import Path
ws = json.loads(Path("results/m4_e5_winners.json").read_text())
ws = sorted(ws, key=lambda w: (w.get("mean_decode_ms") is None, w.get("mean_decode_ms") or 1e9))
print(ws[0]["label"] if ws else "")
PY
)
    echo "best winner: $BEST"
    case "$BEST" in
      compile_old_cachepos)
        run env TENSCUDA_COMPILE=1 "$PY" scripts/m4_replay_probe.py HuggingFaceTB/SmolLM2-135M-Instruct 32
        ;;
      compile_cpuinit_cachepos|legacy_cpuinit_cachepos)
        LEGACY=0
        [[ "$BEST" == legacy* ]] && LEGACY=1
        run env TENSCUDA_COMPILE=1 TENSCUDA_STATIC_CACHE_CPU_INIT=1 TENSCUDA_COMPILE_LEGACY=$LEGACY \
          "$PY" scripts/m4_replay_probe.py HuggingFaceTB/SmolLM2-135M-Instruct 32
        run env TENSCUDA_COMPILE=1 TENSCUDA_STATIC_CACHE_CPU_INIT=1 TENSCUDA_COMPILE_LEGACY=$LEGACY \
          TENSCUDA_DECODE_PAUSE_MODES=1 \
          "$PY" scripts/m4_replay_probe.py HuggingFaceTB/SmolLM2-135M-Instruct 32
        ;;
      compile_cpuinit_posids|legacy_cpuinit_posids)
        echo "winner uses position_ids path; probe covers it — generate.py still cache_position-based"
        ;;
    esac
  else
    echo "No E5 token-matching compile unlock yet."
  fi
fi

echo "=== overnight M4 unlock done $(date -Is) ==="
