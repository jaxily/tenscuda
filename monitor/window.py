"""Main window — stacked Resources-style charts for the Tenstorrent card."""
from __future__ import annotations

from monitor._gi_cairo_bootstrap import ensure as _ensure_gi_cairo

_ensure_gi_cairo()

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from monitor.activity import format_workload, latest, latest_decode
from monitor.charts import AreaChart
from monitor.sysfs import snapshot


class MonitorWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(title="TensCUDA Monitor", default_width=720, default_height=720, **kwargs)

        self._meta = Gtk.Label(xalign=0, wrap=True)
        self._meta.add_css_class("dim-label")
        self._work = Gtk.Label(label="Idle", xalign=0)
        self._work.add_css_class("heading")

        # Colors close to GNOME System Monitor Resources (calm, distinct)
        self._tok = AreaChart(
            "Decode speed", "tok/s", (0.55, 0.35, 0.85),
            format_value=lambda v: "—" if v is None else f"{v:.1f} tok/s",
            y_min=0,
        )
        self._temp = AreaChart(
            "Temperature", "°C", (0.86, 0.28, 0.22),
            format_value=lambda v: "—" if v is None else f"{v:.0f} °C",
            y_min=20, y_max=100,
        )
        self._power = AreaChart(
            "Power", "W", (0.95, 0.55, 0.15),
            format_value=lambda v: "—" if v is None else f"{v:.0f} W",
            y_min=0,
        )
        self._clock = AreaChart(
            "AI clock", "MHz", (0.20, 0.55, 0.90),
            format_value=lambda v: "—" if v is None else f"{v:.0f} MHz",
            y_min=0,
        )
        self._ram = AreaChart(
            "Memory", "GiB", (0.30, 0.70, 0.40),
            format_value=self._fmt_ram,
            y_min=0,
        )

        charts = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        charts.set_margin_top(8)
        charts.set_margin_bottom(16)
        charts.set_margin_start(16)
        charts.set_margin_end(16)
        for w in (self._meta, self._work, self._tok, self._temp, self._power, self._clock, self._ram):
            charts.append(w)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_child(charts)

        header = Adw.HeaderBar()
        title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        logo = Gtk.Image.new_from_icon_name("dev.tenscuda.Monitor")
        logo.set_pixel_size(28)
        title = Gtk.Label(label="TensCUDA Monitor")
        title.add_css_class("heading")
        title_box.append(logo)
        title_box.append(title)
        header.set_title_widget(title_box)
        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(header)
        toolbar.set_content(scroll)
        self.set_content(toolbar)

        self._tick()
        GLib.timeout_add_seconds(1, self._tick)

    @staticmethod
    def _fmt_ram(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v:.1f} GiB"

    def _tick(self) -> bool:
        snap = snapshot()
        card = snap.get("card_type") or "No card"
        bits = [
            card,
            snap.get("architecture") or None,
            f"FW {snap['fw_bundle']}" if snap.get("fw_bundle") else None,
            f"PCIe {snap['pcie']}" if snap.get("pcie") else None,
            f"KMD {snap['kmd']}" if snap.get("kmd") else None,
            "device ready" if snap.get("device_present") else "device missing",
        ]
        if snap.get("ram_total_gib") is not None:
            bits.append(f"Host RAM {snap.get('ram_used_gib') or '—'} / {snap['ram_total_gib']} GiB")
        self._meta.set_label("  ·  ".join(b for b in bits if b))

        self._work.set_label(format_workload(latest()))

        decode = latest_decode()
        # Idle → 0 so the chart drops like System Monitor network; live → rate + model.
        if decode is None:
            self._tok.push(0.0)
            self._tok.set_value_text("—")
        else:
            tok_s = decode["tok_s"]
            self._tok.push(tok_s)
            label = f"{tok_s:.1f} tok/s"
            if decode.get("model"):
                label = f"{label}  ·  {decode['model']}"
            if decode.get("params"):
                label = f"{label} ({decode['params']})"
            self._tok.set_value_text(label)
            if self._tok.y_max_fixed is None or tok_s > self._tok.y_max_fixed * 0.9:
                self._tok.y_max_fixed = max(tok_s * 1.25, 10.0)

        self._temp.push(snap.get("temp_c"))
        self._power.push(snap.get("power_w"))
        self._clock.push(snap.get("aiclk_mhz"))
        used = snap.get("ram_used_gib")
        total = snap.get("ram_total_gib")
        if total is not None and self._ram.y_max_fixed is None:
            self._ram.y_max_fixed = max(total * 1.02, 1.0)
        self._ram.push(used)
        if used is not None and total is not None:
            self._ram.set_value_text(f"{used:.1f} / {total:.1f} GiB")
        return True
