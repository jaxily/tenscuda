"""python -m monitor  |  python -m monitor --tty"""
from __future__ import annotations

import sys


_TTY_FLAGS = {"--tty", "--term", "--terminal", "-t"}
_TTY_CMDS = {"tty", "term", "terminal"}


def _wants_tty(argv: list[str]) -> bool:
    if any(a in _TTY_FLAGS for a in argv[1:]):
        return True
    return len(argv) > 1 and argv[1] in _TTY_CMDS


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if _wants_tty(argv):
        cleaned = [argv[0]]
        for a in argv[1:]:
            if a in _TTY_FLAGS or a in _TTY_CMDS:
                continue
            cleaned.append(a)
        from monitor.term import main as term_main

        return term_main(cleaned[1:])
    from monitor.app import main as gui_main

    return gui_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
