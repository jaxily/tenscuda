"""Load user-local gi._gi_cairo when python3-gi-cairo is not system-installed."""
from __future__ import annotations

import os
import sys


def ensure() -> None:
    try:
        from gi import _gi_cairo  # noqa: F401
        return
    except ImportError:
        pass

    candidates = [
        os.path.expanduser("~/.local/lib/python3/dist-packages"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages"),
    ]
    for root in candidates:
        so_dir = os.path.join(root, "gi")
        if not os.path.isdir(so_dir):
            continue
        if not any(name.startswith("_gi_cairo") for name in os.listdir(so_dir)):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        import gi
        from pkgutil import extend_path

        gi.__path__ = extend_path(list(gi.__path__), gi.__name__)
        try:
            from gi import _gi_cairo  # noqa: F401
            return
        except ImportError:
            continue
