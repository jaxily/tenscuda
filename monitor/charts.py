"""System Monitor–style filled area charts (GTK4 DrawingArea + Cairo)."""
from __future__ import annotations

import collections
from typing import Callable

from monitor._gi_cairo_bootstrap import ensure as _ensure_gi_cairo

_ensure_gi_cairo()

import cairo  # noqa: E402
import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402


class AreaChart(Gtk.Box):
    """Title + current value header, then a scrolling filled line chart."""

    def __init__(
        self,
        title: str,
        unit: str,
        color: tuple[float, float, float],
        *,
        history: int = 60,
        format_value: Callable[[float | None], str] | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        **kwargs,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4, **kwargs)
        self.title = title
        self.unit = unit
        self.color = color
        self.y_min_fixed = y_min
        self.y_max_fixed = y_max
        self.format_value = format_value or (lambda v: "—" if v is None else f"{v:.0f} {unit}")
        self.samples: collections.deque[float | None] = collections.deque(
            [None] * history, maxlen=history
        )

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        header.set_hexpand(True)
        self._title_lbl = Gtk.Label(label=title, xalign=0)
        self._title_lbl.add_css_class("heading")
        self._value_lbl = Gtk.Label(label="—", xalign=1)
        self._value_lbl.set_hexpand(True)
        self._value_lbl.add_css_class("dim-label")
        header.append(self._title_lbl)
        header.append(self._value_lbl)

        self._draw = Gtk.DrawingArea()
        self._draw.set_content_height(140)
        self._draw.set_hexpand(True)
        self._draw.set_draw_func(self._on_draw)

        self.append(header)
        self.append(self._draw)

    def push(self, value: float | None) -> None:
        self.samples.append(value)
        self.set_value_text(self.format_value(value))
        self._draw.queue_draw()

    def set_value_text(self, text: str) -> None:
        self._value_lbl.set_label(text)

    def set_title(self, title: str) -> None:
        self.title = title
        self._title_lbl.set_label(title)

    def _on_draw(self, _area: Gtk.DrawingArea, cr: cairo.Context, width: int, height: int) -> None:
        style = self.get_style_context()
        ok, fg = style.lookup_color("view_fg_color")
        if not ok:
            ok, fg = style.lookup_color("theme_fg_color")
        grid_a = 0.12
        if ok:
            cr.set_source_rgba(fg.red, fg.green, fg.blue, grid_a)
        else:
            cr.set_source_rgba(0.5, 0.5, 0.5, grid_a)

        # background is provided by the window; draw light grid
        pad_l, pad_r, pad_t, pad_b = 4, 4, 4, 4
        w = max(1, width - pad_l - pad_r)
        h = max(1, height - pad_t - pad_b)
        for i in range(5):
            y = pad_t + h * i / 4
            cr.move_to(pad_l, y)
            cr.line_to(pad_l + w, y)
        cr.set_line_width(1)
        cr.stroke()

        vals = [v for v in self.samples if v is not None]
        if not vals:
            return
        ymin = self.y_min_fixed if self.y_min_fixed is not None else min(vals)
        ymax = self.y_max_fixed if self.y_max_fixed is not None else max(vals)
        if ymax <= ymin:
            ymax = ymin + 1.0
        # headroom so flat lines aren't glued to the top
        if self.y_max_fixed is None:
            span = ymax - ymin
            ymax += span * 0.08 + 0.5
            if self.y_min_fixed is None and ymin > 0:
                ymin = max(0.0, ymin - span * 0.05)

        r, g, b = self.color
        n = len(self.samples)
        points: list[tuple[float, float | None]] = []
        for i, v in enumerate(self.samples):
            x = pad_l + (w * i / max(1, n - 1))
            if v is None:
                points.append((x, None))
            else:
                y = pad_t + h * (1.0 - (v - ymin) / (ymax - ymin))
                points.append((x, y))

        # filled area
        started = False
        cr.set_source_rgba(r, g, b, 0.28)
        for x, y in points:
            if y is None:
                if started:
                    cr.line_to(x, pad_t + h)
                    cr.close_path()
                    cr.fill()
                    started = False
                continue
            if not started:
                cr.move_to(x, pad_t + h)
                cr.line_to(x, y)
                started = True
            else:
                cr.line_to(x, y)
        if started:
            last_x = points[-1][0]
            cr.line_to(last_x, pad_t + h)
            cr.close_path()
            cr.fill()

        # stroke
        cr.set_source_rgb(r, g, b)
        cr.set_line_width(2.0)
        started = False
        for x, y in points:
            if y is None:
                started = False
                continue
            if not started:
                cr.move_to(x, y)
                started = True
            else:
                cr.line_to(x, y)
        cr.stroke()
