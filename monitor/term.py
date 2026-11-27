"""Terminal (TTY) TensCUDA Monitor — same sources as the GTK UI, no GUI deps."""
from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import sys
import time
from typing import Any

from monitor.activity import format_workload, latest_chart, latest_workload
from monitor.sysfs import snapshot

_BLOCKS = " ▁▂▃▄▅▆▇█"
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# Palette aligned with monitor.window chart hues.
_C_RESET = "\033[0m"
_C_BOLD = "\033[1m"
_C_DIM = "\033[2m"
_BG_PANEL = (18, 20, 28)
_FG_MUTED = (120, 128, 145)
_FG_SOFT = (170, 178, 195)
_FG_BRIGHT = (236, 240, 248)
_ACCENT = (110, 168, 255)
_OK = (72, 200, 120)
_WARN = (242, 180, 60)
_BAD = (235, 80, 70)
_METRIC = {
    "workload": (155, 105, 235),
    "temp": (235, 90, 75),
    "power": (245, 155, 55),
    "clock": (70, 160, 245),
    "ram": (70, 195, 120),
}


def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def _paint(text: str, *codes: str, on: bool) -> str:
    if not on or not text:
        return text
    return f"{''.join(codes)}{text}{_C_RESET}"


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _pad_visible(s: str, width: int, align: str = "left") -> str:
    pad = max(0, width - _visible_len(s))
    if align == "right":
        return " " * pad + s
    if align == "center":
        left = pad // 2
        return " " * left + s + " " * (pad - left)
    return s + " " * pad


def _truncate_visible(s: str, width: int) -> str:
    if _visible_len(s) <= width:
        return s
    plain = _ANSI_RE.sub("", s)
    if width <= 1:
        return plain[:width]
    return plain[: max(0, width - 1)] + "…"


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    flag = os.environ.get("TENSCUDA_MONITOR_COLOR", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on", "always"}:
        return True
    return sys.stdout.isatty()


def _temp_color(temp: float | None) -> tuple[int, int, int]:
    if temp is None:
        return _METRIC["temp"]
    if temp < 55:
        return _lerp((70, 180, 220), (70, 195, 120), (temp - 30) / 25)
    if temp < 75:
        return _lerp((70, 195, 120), _WARN, (temp - 55) / 20)
    return _lerp(_WARN, _BAD, min(1.0, (temp - 75) / 20))


def _gradient_bar(
    frac: float,
    width: int,
    *,
    color: tuple[int, int, int],
    on: bool,
) -> str:
    frac = max(0.0, min(1.0, frac))
    filled_f = frac * width
    filled = int(filled_f)
    partial = filled_f - filled
    dark = _lerp(color, (20, 22, 30), 0.72)
    track = _lerp(color, (35, 38, 48), 0.85)
    cells: list[str] = []
    for i in range(width):
        if i < filled:
            # Brighten toward the leading edge.
            t = 0.35 + 0.65 * (i / max(1, width - 1))
            c = _lerp(dark, color, t)
            ch = "█"
        elif i == filled and partial > 0.12:
            c = _lerp(dark, color, 0.55 + 0.45 * partial)
            ch = "▌" if partial < 0.66 else "█"
        else:
            c = track
            ch = "░"
        if on:
            cells.append(f"{_rgb(*c)}{ch}")
        else:
            cells.append(ch)
    return "".join(cells) + (_C_RESET if on else "")


def _spark(
    samples: collections.deque[float | None],
    width: int = 36,
    *,
    color: tuple[int, int, int] | None = None,
    on: bool = False,
) -> str:
    vals = [v for v in samples if v is not None]
    if not vals:
        return " " * width
    lo = min(vals)
    hi = max(vals)
    span = hi - lo if hi > lo else 1.0
    # Prefer real samples; right-align so history grows toward the leading edge.
    real = [v for v in samples if v is not None][-width:]
    pad = width - len(real)
    out: list[str] = [" "] * pad
    for i, v in enumerate(real):
        idx = int((v - lo) / span * (len(_BLOCKS) - 1))
        ch = _BLOCKS[max(0, min(len(_BLOCKS) - 1, idx))]
        if on and color is not None:
            age = (pad + i) / max(1, width - 1)
            c = _lerp(_lerp(color, (40, 44, 55), 0.7), color, 0.2 + 0.8 * age)
            out.append(f"{_rgb(*c)}{ch}")
        else:
            out.append(ch)
    return "".join(out) + (_C_RESET if on and color is not None else "")


def _fmt_meta_bits(snap: dict[str, Any]) -> list[str]:
    bits = [
        snap.get("card_type") or "No card",
        snap.get("architecture") or None,
        f"FW {snap['fw_bundle']}" if snap.get("fw_bundle") else None,
        f"PCIe {snap['pcie']}" if snap.get("pcie") else None,
        f"KMD {snap['kmd']}" if snap.get("kmd") else None,
    ]
    return [b for b in bits if b]


def _fmt_meta(snap: dict[str, Any]) -> str:
    bits = _fmt_meta_bits(snap)
    bits.append("device ready" if snap.get("device_present") else "device missing")
    if snap.get("ram_total_gib") is not None:
        bits.append(f"Host RAM {snap.get('ram_used_gib') or '—'} / {snap['ram_total_gib']} GiB")
    return "  ·  ".join(bits)


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


def _badge(text: str, fg: tuple[int, int, int], on: bool) -> str:
    body = f" {text} "
    if not on:
        return f"[{text}]"
    soft_bg = _lerp(fg, _BG_PANEL, 0.78)
    return f"{_bg(*soft_bg)}{_rgb(*fg)}{_C_BOLD}{body}{_C_RESET}"


def _box_line(inner: str, width: int, *, on: bool, edge: str = "│") -> str:
    content = _pad_visible(_truncate_visible(inner, width - 4), width - 4)
    if not on:
        return f"{edge} {content} {edge}"
    return (
        f"{_rgb(*_FG_MUTED)}{edge}{_C_RESET} "
        f"{content} "
        f"{_rgb(*_FG_MUTED)}{edge}{_C_RESET}"
    )


def _hrule(width: int, *, on: bool, left: str = "├", mid: str = "─", right: str = "┤") -> str:
    body = mid * max(0, width - 2)
    line = f"{left}{body}{right}"
    return _paint(line, _rgb(*_FG_MUTED), on=on)


def render_frame(
    snap: dict[str, Any],
    work: str,
    chart: dict[str, Any] | None,
    history: dict[str, collections.deque[float | None]],
    *,
    cols: int = 80,
    color: bool = False,
) -> str:
    """Build one full-screen frame (no ANSI clear). Pure for tests."""
    width = max(56, min(100, cols))
    inner = width - 4
    bar_w = max(24, inner)
    spark_w = max(24, inner)

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

    ready = bool(snap.get("device_present"))
    idle = (work or "").strip().lower() == "idle"
    status_badge = _badge("READY" if ready else "MISSING", _OK if ready else _BAD, color)
    work_badge = _badge("IDLE" if idle else "LIVE", _FG_SOFT if idle else _ACCENT, color)

    brand = _paint("◆ TensCUDA Monitor", _rgb(*_ACCENT), _C_BOLD, on=color) + _paint(
        "  tty", _rgb(*_FG_MUTED), on=color
    )
    brand_line = f"{brand}  {status_badge}  {work_badge}"

    meta_bits = _fmt_meta_bits(snap)
    meta_plain = _truncate_visible("  ·  ".join(meta_bits) if meta_bits else "—", inner)
    meta = _paint(meta_plain, _rgb(*_FG_MUTED), on=color)

    work_line = (
        _paint("Workload", _rgb(*_FG_MUTED), on=color)
        + "  "
        + _paint(work, _rgb(*_FG_BRIGHT), _C_BOLD, on=color)
    )

    def metric_block(
        name: str,
        value: str,
        sample: float | None,
        scale: float,
        key: str,
        hue: tuple[int, int, int] | None = None,
    ) -> list[str]:
        hue = hue or _METRIC[key]
        frac = 0.0 if sample is None or scale <= 0 else float(sample) / scale
        name_s = _paint(name, _rgb(*hue), _C_BOLD, on=color)
        value_s = _paint(value, _rgb(*_FG_BRIGHT), _C_BOLD, on=color)
        # Right-align the live value for a dashboard feel.
        head = _pad_visible(name_s, 16) + _pad_visible(
            _truncate_visible(value_s, max(8, inner - 16)), max(8, inner - 16), align="right"
        )
        bar = _gradient_bar(frac, bar_w, color=hue, on=color)
        spark = _spark(history[key], spark_w, color=hue, on=color)
        return [head, bar, spark]

    ram_scale = float(ram_total or 1.0)
    ram_val = (
        f"{ram_used:.1f} / {ram_total:.1f} GiB"
        if ram_used is not None and ram_total is not None
        else "—"
    )

    blocks: list[list[str]] = [
        metric_block(c_title, c_val, c_num, c_max, "workload"),
        metric_block(
            "Temperature",
            "—" if temp is None else f"{temp:.0f} °C",
            temp,
            100.0,
            "temp",
            hue=_temp_color(temp),
        ),
        metric_block(
            "Power",
            "—" if power is None else f"{power:.0f} W",
            power,
            max(power or 0.0, 200.0),
            "power",
        ),
        metric_block(
            "AI clock",
            "—" if clock is None else f"{clock:.0f} MHz",
            clock,
            max(clock or 0.0, 1500.0),
            "clock",
        ),
        metric_block("Host RAM", ram_val, ram_used, ram_scale, "ram"),
    ]

    top = _paint("╭" + "─" * (width - 2) + "╮", _rgb(*_FG_MUTED), on=color)
    bot = _paint("╰" + "─" * (width - 2) + "╯", _rgb(*_FG_MUTED), on=color)
    lines = [
        top,
        _box_line(brand_line, width, on=color),
        _box_line(meta, width, on=color),
        _hrule(width, on=color),
        _box_line(work_line, width, on=color),
        _hrule(width, on=color),
    ]
    for i, block in enumerate(blocks):
        for row in block:
            lines.append(_box_line(row, width, on=color))
        if i < len(blocks) - 1:
            hair = _paint("·" * (width - 4), _rgb(45, 50, 62), on=color)
            lines.append(_box_line(hair, width, on=color))
    foot = _paint("q quit", _rgb(*_ACCENT), on=color) + _paint(
        "  ·  Ctrl-C  ·  refresh 1s", _rgb(*_FG_MUTED), on=color
    )
    lines.append(_hrule(width, on=color))
    lines.append(_box_line(foot, width, on=color))
    lines.append(bot)
    return "\n".join(lines)


def new_history(n: int = 60) -> dict[str, collections.deque[float | None]]:
    return {
        key: collections.deque([None] * n, maxlen=n)
        for key in ("workload", "temp", "power", "clock", "ram")
    }


def run(interval_s: float = 1.0) -> int:
    history = new_history()
    hide = "\033[?25l"
    show = "\033[?25h"
    alt_on = "\033[?1049h"
    alt_off = "\033[?1049l"
    clear = "\033[H\033[2J"
    use_tty = sys.stdout.isatty()
    if use_tty:
        sys.stdout.write(alt_on + hide)
        sys.stdout.flush()
    try:
        while True:
            snap = snapshot()
            work = format_workload(latest_workload())
            chart = latest_chart()
            cols = shutil.get_terminal_size((80, 24)).columns
            frame = render_frame(
                snap, work, chart, history, cols=cols, color=color_enabled()
            )
            if use_tty:
                sys.stdout.write(clear + frame + "\n")
            else:
                sys.stdout.write(frame + "\n\n")
            sys.stdout.flush()
            if _poll_quit(interval_s):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if use_tty:
            sys.stdout.write(show + alt_off)
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
