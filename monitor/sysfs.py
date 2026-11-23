"""Read Tenstorrent card + host stats from sysfs / proc (no torch)."""
from __future__ import annotations

import glob
import os
from typing import Any


def _read(path: str, default: str | None = None) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default


def _read_int(path: str) -> int | None:
    raw = _read(path)
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except ValueError:
        return None


def cards() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*")):
        dev = os.path.join(d, "device")
        speed = (_read(os.path.join(dev, "current_link_speed"), "?") or "?").replace(" PCIe", "")
        width = _read(os.path.join(dev, "current_link_width"), "?") or "?"
        aiclk = _read_int(os.path.join(d, "tt_aiclk"))
        out.append({
            "type": _read(os.path.join(d, "tt_card_type")),
            "fw_bundle": _read(os.path.join(d, "tt_fw_bundle_ver")),
            "m3_app_fw": _read(os.path.join(d, "tt_m3app_fw_ver")),
            "aiclk_mhz": float(aiclk) if aiclk is not None else None,
            "pcie": f"{speed} x{width}",
            "path": d,
        })
    return out


def _hwmon_dirs() -> list[str]:
    return sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/device/hwmon/hwmon*"))


def hwmon_sample() -> dict[str, float | None]:
    """Average temp °C and power W across all card hwmon nodes."""
    temps: list[float] = []
    powers: list[float] = []
    for h in _hwmon_dirs():
        t = _read_int(os.path.join(h, "temp1_input"))
        p = _read_int(os.path.join(h, "power1_input"))
        if t is not None:
            temps.append(t / 1000.0)
        if p is not None:
            powers.append(p / 1e6)
    return {
        "temp_c": (sum(temps) / len(temps)) if temps else None,
        "power_w": (sum(powers) / len(powers)) if powers else None,
    }


def host_memory() -> dict[str, float | None]:
    """Host RAM used / total in GiB from /proc/meminfo."""
    total_kib = avail_kib = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total_kib = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kib = int(line.split()[1])
    except OSError:
        return {"used_gib": None, "total_gib": None}
    if total_kib is None:
        return {"used_gib": None, "total_gib": None}
    total = total_kib / 2**20
    if avail_kib is None:
        return {"used_gib": None, "total_gib": round(total, 2)}
    used = (total_kib - avail_kib) / 2**20
    return {"used_gib": round(used, 2), "total_gib": round(total, 2)}


def kmd_version() -> str | None:
    return _read("/sys/module/tenstorrent/version")


def device_present() -> bool:
    return (
        os.path.exists("/dev/tenstorrent0")
        or os.path.exists("/dev/tenstorrent/0")
        or bool(glob.glob("/dev/tenstorrent/*"))
    )


def architecture_label(card_type: str | None) -> str:
    if not card_type:
        return "—"
    ct = card_type.lower()
    if "blackhole" in ct or "p150" in ct or "bh" in ct:
        return "Blackhole · Tensix"
    if "wormhole" in ct or "n150" in ct or "n300" in ct:
        return "Wormhole · Tensix"
    return card_type


def snapshot() -> dict[str, Any]:
    """One poll of everything the UI needs from the host/card."""
    c = cards()
    first = c[0] if c else {}
    hw = hwmon_sample()
    mem = host_memory()
    aiclk = first.get("aiclk_mhz")
    if aiclk is None and c:
        # refresh clock from first card path if present as string earlier
        pass
    return {
        "card_count": len(c),
        "card_type": first.get("type"),
        "architecture": architecture_label(first.get("type")),
        "fw_bundle": first.get("fw_bundle"),
        "m3_app_fw": first.get("m3_app_fw"),
        "pcie": first.get("pcie"),
        "aiclk_mhz": aiclk,
        "temp_c": hw["temp_c"],
        "power_w": hw["power_w"],
        "kmd": kmd_version(),
        "device_present": device_present(),
        "ram_used_gib": mem["used_gib"],
        "ram_total_gib": mem["total_gib"],
    }
