#!/usr/bin/env bash
# Shared card hygiene for native tt-metal / vLLM-TT runs (one job on CHIP_IN_USE).
# shellcheck disable=SC2034
set -euo pipefail

_card_hygiene_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  echo "$here"
}

# Stop TensCUDA matrix / leftover vLLM serve processes before native fast paths.
card_hygiene() {
  local quiet="${1:-0}"
  pkill -f 'scripts/matrix.py' 2>/dev/null || true
  pkill -f 'run_tens_' 2>/dev/null || true
  pkill -f 'server_example_tt.py' 2>/dev/null || true
  sleep 1

  if pgrep -af 'pjrt-plugin|tenscuda.*matrix' >/dev/null 2>&1; then
    if [[ "$quiet" -eq 0 ]]; then
      echo "WARN: TensCUDA/matrix may still be on the card — pgrep -af matrix" >&2
    fi
    return 1
  fi
  return 0
}

wait_vllm_health() {
  local port="${1:-8000}"
  local tries="${2:-180}"
  local delay_s="${3:-2}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_s"
  done
  return 1
}

vllm_serve_log_has_engine_dead() {
  local log="${1:-/tmp/vllm_serve_8b.log}"
  [[ -f "$log" ]] || return 1
  grep -qE 'EngineDeadError|engine dead|Engine has died' "$log" 2>/dev/null
}