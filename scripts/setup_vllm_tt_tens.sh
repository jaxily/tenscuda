#!/usr/bin/env bash
# One-shot vLLM-TT install on `tens` (separate from ~/tenscuda/.venv).
# Prereq: ~/tt-metal built, python_env activated per TT_METAL_FASTPATH.md.
set -euo pipefail

TT_METAL="${TT_METAL_HOME:-$HOME/tt-metal}"
PY_ENV="${PYTHON_ENV_DIR:-$TT_METAL/python_env}"

if [[ ! -d "$TT_METAL" ]]; then
  echo "Missing TT_METAL at $TT_METAL" >&2
  exit 1
fi

export TT_METAL_HOME="$TT_METAL"
export PYTHONPATH="$TT_METAL"
# shellcheck disable=SC1091
source "$PY_ENV/bin/activate"

PLUGIN="${VLLM_TT_PLUGIN_DIR:-$HOME/vllm-tt-plugin}"
if [[ ! -d "$PLUGIN/.git" ]]; then
  git clone https://github.com/tenstorrent/vllm-tt-plugin.git "$PLUGIN"
fi

cd "$PLUGIN"
git fetch --depth 1 origin main 2>/dev/null || true
git pull --ff-only 2>/dev/null || true

if [[ ! -f docs/install-vllm-tt.sh ]]; then
  echo "install-vllm-tt.sh missing in $PLUGIN" >&2
  exit 1
fi

# Network + uv; may take 15–30 min first time.
bash docs/install-vllm-tt.sh

echo "OK: vLLM TT in $PY_ENV — see vllm-tt-plugin/README.md for serve examples"