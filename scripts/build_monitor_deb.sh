#!/usr/bin/env bash
# Build dist/tenscuda-monitor_<ver>_all.deb from the repo tree.
# Run on Ubuntu/Debian (needs dpkg-deb). Safe to run from macOS checkout via ssh to the card host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_NAME="tenscuda-monitor"
VERSION="$(awk -F': ' '/^Version:/{print $2; exit}' "$ROOT/packaging/$PKG_NAME/DEBIAN/control")"
ARCH="$(awk -F': ' '/^Architecture:/{print $2; exit}' "$ROOT/packaging/$PKG_NAME/DEBIAN/control")"
OUT_DIR="$ROOT/dist"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/tenscuda-monitor-deb.XXXXXX")"
DEB="$OUT_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

echo "Staging $PKG_NAME $VERSION ..."
mkdir -p "$STAGE"
cp -a "$ROOT/packaging/$PKG_NAME/." "$STAGE/"

# Python package
mkdir -p "$STAGE/usr/lib/tenscuda-monitor"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'icons' \
  "$ROOT/monitor/" "$STAGE/usr/lib/tenscuda-monitor/monitor/"

# Icons
mkdir -p "$STAGE/usr/share/icons"
cp -a "$ROOT/monitor/icons/hicolor" "$STAGE/usr/share/icons/"

# Desktop entry (Exec already = uptens)
mkdir -p "$STAGE/usr/share/applications"
cp "$ROOT/monitor/uptens.desktop" "$STAGE/usr/share/applications/"

# Launcher permissions + DEBIAN metadata
chmod 755 "$STAGE/usr/bin/uptens"
chmod 755 "$STAGE/DEBIAN"
chmod 644 "$STAGE/DEBIAN/control"

# Installed-Size in KiB
size_kib=$(du -sk "$STAGE" | awk '{print $1}')
if grep -q '^Installed-Size:' "$STAGE/DEBIAN/control"; then
  sed -i.bak "s/^Installed-Size:.*/Installed-Size: $size_kib/" "$STAGE/DEBIAN/control" && rm -f "$STAGE/DEBIAN/control.bak"
else
  printf 'Installed-Size: %s\n' "$size_kib" >> "$STAGE/DEBIAN/control"
fi

mkdir -p "$OUT_DIR"
rm -f "$DEB"
dpkg-deb --root-owner-group --build "$STAGE" "$DEB"
echo "Built $DEB"
dpkg-deb -I "$DEB"
dpkg-deb -c "$DEB" | head -40
