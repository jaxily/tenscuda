#!/usr/bin/env bash
# Install TensCUDA Monitor as a normal Ubuntu app (Activities + CLI).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$ROOT/scripts/tenscuda-monitor" "$ROOT/scripts/install_monitor_on_host.sh"

# CLI command on PATH
mkdir -p "$HOME/.local/bin"
ln -sfn "$ROOT/scripts/tenscuda-monitor" "$HOME/.local/bin/tenscuda-monitor"

# Ensure ~/.local/bin is on PATH for future shells
for rc in "$HOME/.bashrc" "$HOME/.profile"; do
  if [[ -f "$rc" ]] && ! grep -q '\$HOME/.local/bin' "$rc" 2>/dev/null; then
    printf '\n# TensCUDA Monitor\ncase ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac\nexport PATH\n' >> "$rc"
  fi
done
export PATH="$HOME/.local/bin:$PATH"

# App icon (SVG + PNG sizes)
ICON_ROOT="$HOME/.local/share/icons/hicolor"
mkdir -p "$ICON_ROOT"
cp -a "$ROOT/monitor/icons/hicolor/." "$ICON_ROOT/"
gtk-update-icon-cache -f "$ICON_ROOT" 2>/dev/null || true

# Desktop entry (Activities / app grid)
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$APP_DIR"
# Rewrite Exec to absolute path so it works even before PATH is refreshed
sed "s|^Exec=tenscuda-monitor$|Exec=$HOME/.local/bin/tenscuda-monitor|" \
  "$ROOT/monitor/tenscuda-monitor.desktop" > "$APP_DIR/tenscuda-monitor.desktop"
chmod 644 "$APP_DIR/tenscuda-monitor.desktop"
update-desktop-database "$APP_DIR" 2>/dev/null || true

# User-local cairo GI binding if system package missing
if ! python3 -c "from gi import _gi_cairo" 2>/dev/null; then
  TMP=$(mktemp -d)
  (
    cd "$TMP"
    apt-get download python3-gi-cairo >/dev/null
    dpkg-deb -x python3-gi-cairo_*.deb extract
    mkdir -p "$HOME/.local/lib/python3/dist-packages/gi"
    cp extract/usr/lib/python3/dist-packages/gi/_gi_cairo*.so \
      "$HOME/.local/lib/python3/dist-packages/gi/"
  )
  rm -rf "$TMP"
fi

# Refresh GNOME app cache so it appears immediately
if command -v gio >/dev/null; then
  gio mime x-scheme-handler/tenscuda >/dev/null 2>&1 || true
fi

echo "Installed:"
echo "  Activities search: TensCUDA Monitor"
echo "  Terminal:          tenscuda-monitor"
echo "  gtk-launch:        gtk-launch tenscuda-monitor"
