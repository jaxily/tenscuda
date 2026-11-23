#!/usr/bin/env bash
# Install Docker (needs sudo once), then serve MiniMax-H3 on the P150 via tt-model.
# Run on tens:  bash ~/tenscuda/scripts/setup_minimax_h3_tens.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME"

echo "== MiniMax-H3 on P150 =="
echo "Bundle: jashansinghTT/minimax-h3-blackhole (profile p150)"
echo "Needs: ~144 GiB weights + ~26 GiB cache, Docker, card at /dev/tenstorrent/0"
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
exec tt serve jashansinghTT/minimax-h3-blackhole --profile p150
