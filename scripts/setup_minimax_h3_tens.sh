#!/usr/bin/env bash
# Install Docker (needs sudo once), then serve MiniMax-H3 on the P150 via tt-model.
# Run on tens:  bash ~/tenscuda/scripts/setup_minimax_h3_tens.sh
#
# Max-video defaults for this card (measured in the bundle's config_defaults):
#   - 124 frames @ 768p ≈ 5.2 s / request (NFE 4 ≈ 158–162 s warm e2e)
#   - H3_QUALITY=8 switches to the 8-step Turbo adapter (better CLIP, ~2× denoise)
# Full undistilled 49-step H3 is NOT served on P150 — needs a Galaxy mesh.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME"

PROFILE="${H3_PROFILE:-p150}"
QUALITY="${H3_QUALITY:-4}"   # 4 (default, longest throughput) or 8 (higher fidelity)

case "$QUALITY" in
  4)
    export H3_TURBO_FILE="${H3_TURBO_FILE:-minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors}"
    STEPS=4
    ;;
  8)
    export H3_TURBO_FILE="${H3_TURBO_FILE:-minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors}"
    STEPS=8
    ;;
  *)
    echo "H3_QUALITY must be 4 or 8 (got $QUALITY)" >&2
    exit 1
    ;;
esac

# Push the served working point to the measured frame ceiling (bundle default is 73 / 3.0 s).
export H3_FRAMES_MAX="${H3_FRAMES_MAX:-124}"
export H3_STEPS_MAX="${H3_STEPS_MAX:-50}"
export H3_DEFAULTS="${H3_DEFAULTS:-{\"width\":1344,\"height\":768,\"num_frames\":124,\"steps\":${STEPS}}}"
export H3_PROMPT_CACHE="${H3_PROMPT_CACHE:-1}"
export H3_WARMUP="${H3_WARMUP:-1}"
export H3_OUT_KEEP="${H3_OUT_KEEP:-32}"

echo "== MiniMax-H3 on P150 (max video) =="
echo "Bundle:  jashansinghTT/minimax-h3-blackhole (profile $PROFILE)"
echo "Quality: NFE $STEPS  turbo=$H3_TURBO_FILE"
echo "Default: 1344x768, 124 frames (~5.2 s), steps=$STEPS"
echo "Needs:   ~144 GiB weights + ~26 GiB cache, Docker, card at /dev/tenstorrent/0"
echo "Longer:  chain fl2va clips (scripts/minimax_chain_reel.py --frames 124)"
echo

if [[ ! -e /dev/tenstorrent/0 && ! -e /dev/tenstorrent ]]; then
  echo "ERROR: no Tenstorrent device node"
  exit 1
fi

if ! command -v tt >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
  uv tool install tenstorrent
fi

if ! command -v docker >/dev/null; then
  echo "Installing Docker (sudo)…"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo
  echo "Docker installed. Log out/in (or: newgrp docker), then re-run this script."
  exit 0
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker installed but this shell can't talk to the daemon."
  echo "Run:  newgrp docker"
  echo "Then re-run this script."
  exit 1
fi

mkdir -p "$HOME/videos/minimax_h3"
echo "Serving MiniMax-H3 (first boot downloads weights + compiles kernels — long)…"
echo "API will be on http://localhost:20000"
exec tt serve jashansinghTT/minimax-h3-blackhole --profile "$PROFILE"
