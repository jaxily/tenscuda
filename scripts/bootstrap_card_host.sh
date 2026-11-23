#!/usr/bin/env bash
# One-time card-host setup for Ubuntu (needs sudo). Safe to re-run after reboot.
set -euo pipefail

KMD_VER="${TENSTorrent_KMD_VERSION:-2.11.0}"
DEB_URL="https://github.com/tenstorrent/tt-kmd/releases/download/ttkmd-${KMD_VER}/tenstorrent-dkms_${KMD_VER}_all.deb"

echo "== Tenstorrent P150 host bootstrap (KMD ${KMD_VER}) =="

if ! command -v wget >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y wget
fi

if ! dpkg -l tenstorrent-dkms 2>/dev/null | grep -q "^ii"; then
  tmp=$(mktemp -d)
  wget -q -O "$tmp/tenstorrent-dkms.deb" "$DEB_URL"
  sudo apt-get update -qq
  sudo apt-get install -y linux-headers-"$(uname -r)" build-essential dkms || true
  sudo dpkg -i "$tmp/tenstorrent-dkms.deb" || sudo apt-get install -f -y
  rm -rf "$tmp"
fi

if mokutil --sb-state 2>/dev/null | grep -qi enabled; then
  echo ""
  echo "ERROR: Secure Boot is enabled. Unsigned DKMS modules (tenstorrent) are blocked"
  echo "       (modprobe: Key was rejected by service)."
  echo "       Fix: reboot into firmware/BIOS and disable Secure Boot, then run:"
  echo "         sudo modprobe tenstorrent"
  echo "       Or sign the module and enroll a MOK key (see Tenstorrent / Ubuntu DKMS docs)."
  exit 1
fi

if ! lsmod | grep -q tenstorrent; then
  sudo modprobe tenstorrent
fi

if [[ ! -e /dev/tenstorrent/0 && ! -e /dev/tenstorrent0 ]]; then
  echo "WARN: no Tenstorrent char dev — expected /dev/tenstorrent/0 (KMD 2.11+) or /dev/tenstorrent0"
  echo "       check: lsmod tenstorrent; dmesg | tail -30; lspci -vv -s 01:00.0"
  exit 1
fi

echo "KMD:" "$(cat /sys/module/tenstorrent/version 2>/dev/null || echo not-loaded)"
lspci -s 01:00.0 -vv 2>/dev/null | grep -E 'BusMaster|LnkSta|Region' || true
echo
echo "Smoke:"
echo "  cd ~/tenscuda && PYTHONPATH=. .venv/bin/python -c 'import tenscuda; print(tenscuda.is_active)'"