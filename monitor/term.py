"""Terminal (TTY) TensCUDA Monitor — same sources as the GTK UI, no GUI deps."""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys
import time
from typing import Any

from monitor.activity import format_workload, latest_chart, latest_workload
from monitor.sysfs import snapshot

_BLOCKS = " ▁▂▃▄▅▆▇█"
_BAR = "█"
_EMPTY = "░"


def _bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return _BAR * filled + _EMPTY * (width - filled)


def _spark(samples: collections.deque[float | None], width: int = 36) -> str:
    vals = [v for v in samples if v is not None]
    if not vals:
        return " " * width
    lo = min(vals)
    hi = max(vals)
    span = hi - lo if hi > lo else 1.0
    # take the last `width` samples (pad left)
    tail = list(samples)[-width:]
    while len(tail) < width:
        tail.insert(0, None)
    out: list[str] = []
    for v in tail:
        if v is None:
            out.append(" ")
        else:
            idx = int((v - lo) / span * (len(_BLOCKS) - 1))
            out.append(_BLOCKS[max(0, min(len(_BLOCKS) - 1, idx))])
    return "".join(out)


def _fmt_meta(snap: dict[str, Any]) -> str:
    bits = [
        snap.get("card_type") or "No card",
        snap.get("architecture") or None,
        f"FW {snap['fw_bundle']}" if snap.get("fw_bundle") else None,
        f"PCIe {snap['pcie']}" if snap.get("pcie") else None,
        f"KMD {snap['kmd']}" if snap.get("kmd") else None,
        "device ready" if snap.get("device_present") else "device missing",
    ]
    if snap.get("ram_total_gib") is not None:
        bits.append(f"Host RAM {snap.get('ram_used_gib') or '—'} / {snap['ram_total_gib']} GiB")
    return "  ·  ".join(b for b in bits if b)


def _chart_label(chart: dict[str, Any] | None) -> tuple[str, str, float | None, float]:
    """title, value text, numeric sample, scale max for bar."""
    if chart is None:
        return "Decode speed", "—", 0.0, 1.0
    if chart["mode"] == "video":
        pct = float(chart["value"])
        label = f"{pct:.0f}%"
        if chart.get("step") is not None and chart.get("total_steps") is not None:
            label = f"{label}  ·  step {chart['step']}/{chart['total_steps']}"
        if chart.get("model"):
            label = f"{label}  ·  {chart['model']}"
        return "Gen progress", label, pct, 100.0
    tok = float(chart["value"])
    label = f"{tok:.1f} tok/s"
    if chart.get("model"):
        label = f"{label}  ·  {chart['model']}"
    if chart.get("params"):
        label = f"{label} ({chart['params']})"
    scale = max(tok * 1.25, 10.0)
    return "Decode speed", label, tok, scale


def render_frame(
    snap: dict[str, Any],
    work: str,
    chart: dict[str, Any] | None,
    history: dict[str, collections.deque[float | None]],
    *,
    cols: int = 80,
) -> str:
    """Build one full-screen frame (no ANSI clear). Pure for tests."""
    title_w = max(12, min(42, cols - 40))
    spark_w = max(16, min(48, cols - 32))
    bar_w = max(10, min(28, cols - 48))

    c_title, c_val, c_num, c_max = _chart_label(chart)
    history["workload"].append(c_num)

    temp = snap.get("temp_c")
    power = snap.get("power_w")
    clock = snap.get("aiclk_mhz")
    ram_used = snap.get("ram_used_gib")
    ram_total = snap.get("ram_total_gib")
    history["temp"].append(temp)
    history["power"].append(power)
    history["clock"].append(clock)
    history["ram"].append(ram_used)

    def row(name: str, value: str, sample: float | None, scale: float, key: str) -> str:
        frac = 0.0 if sample is None or scale <= 0 else float(sample) / scale
        return (
            f"{name:<14} {value:<{title_w}.{title_w}} "
            f"{_bar(frac, bar_w)} {_spark(history[key], spark_w)}"
        )

    ram_scale = float(ram_total or 1.0)
    ram_val = (
        f"{ram_used:.1f} / {ram_total:.1f} GiB"
        if ram_used is not None and ram_total is not None
        else "—"
    )

    lines = [
        "TensCUDA Monitor (tty)",
        _fmt_meta(snap),
        f"Workload  {work}",
        "",
        row(c_title, c_val, c_num, c_max, "workload"),
        row(
            "Temperature",
            "—" if temp is None else f"{temp:.0f} °C",
            temp,
            100.0,
            "temp",
        ),
        row(
            "Power",
            "—" if power is None else f"{power:.0f} W",
            power,
            max(power or 0.0, 200.0),
            "power",
        ),
        row(
            "AI clock",
            "—" if clock is None else f"{clock:.0f} MHz",
            clock,
            max(clock or 0.0, 1500.0),
            "clock",
        ),
        row("Host RAM", ram_val, ram_used, ram_scale, "ram"),
        "",
        "q quit  ·  Ctrl-C quit  ·  refresh 1s",
    ]
    return "\n".join(lines)


def new_history(n: int = 60) -> dict[str, collections.deque[float | None]]:
    return {
        key: collections.deque([None] * n, maxlen=n)
        for key in ("workload", "temp", "power", "clock", "ram")
    }


def run(interval_s: float = 1.0) -> int:
    history = new_history()
    # Hide cursor; restore on exit.
    hide = "\033[?25l"
    show = "\033[?25h"
    clear = "\033[H\033[2J"
    if sys.stdout.isatty():
        sys.stdout.write(hide)
        sys.stdout.flush()
    try:
        while True:
            snap = snapshot()
            work = format_workload(latest_workload())
            chart = latest_chart()
            cols = shutil.get_terminal_size((80, 24)).columns
            frame = render_frame(snap, work, chart, history, cols=cols)
            if sys.stdout.isatty():
                sys.stdout.write(clear + frame + "\n")
            else:
                sys.stdout.write(frame + "\n\n")
            sys.stdout.flush()
            # Non-blocking quit via termios when TTY.
            if _poll_quit(interval_s):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if sys.stdout.isatty():
            sys.stdout.write(show + "\n")
            sys.stdout.flush()
    return 0


def _poll_quit(interval_s: float) -> bool:
    """Sleep up to interval_s; return True if user pressed q/Q/ESC."""
    if not sys.stdin.isatty():
        time.sleep(interval_s)
        return False
    try:
        import select
        import termios
        import tty
    except ImportError:
        time.sleep(interval_s)
        return False

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        end = time.monotonic() + interval_s
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            r, _, _ = select.select([sys.stdin], [], [], remaining)
            if not r:
                return False
            ch = sys.stdin.read(1)
            if ch in ("q", "Q", "\x1b"):
                return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TensCUDA Monitor (terminal)")
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("TENSCUDA_MONITOR_INTERVAL", "1")),
        help="Refresh interval seconds (default 1)",
    )
    args = parser.parse_args(argv)
    return run(interval_s=max(0.2, args.interval))
