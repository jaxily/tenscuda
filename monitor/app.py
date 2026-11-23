"""Adw application entry for TensCUDA Monitor."""
from __future__ import annotations

import sys

from monitor._gi_cairo_bootstrap import ensure as _ensure_gi_cairo

_ensure_gi_cairo()

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio  # noqa: E402

from monitor.window import MonitorWindow


class MonitorApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="dev.tenscuda.Monitor", flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.set_resource_base_path(None)
        self._window: MonitorWindow | None = None

    def do_activate(self):
        if self._window is None:
            self._window = MonitorWindow(application=self)
            self._window.set_icon_name("dev.tenscuda.Monitor")
        self._window.present()


def main(argv: list[str] | None = None) -> int:
    app = MonitorApp()
    return app.run(argv if argv is not None else sys.argv)
