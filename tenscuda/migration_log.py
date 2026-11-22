"""Habana-style migration logging for cuda -> TT remaps (TENSCUDA_MIGRATION_LOG)."""
from __future__ import annotations

import atexit
import logging
import os
from collections import Counter
from typing import Callable, TypeVar

from .api_status import CUDA_API

log = logging.getLogger("tenscuda.migration")

F = TypeVar("F", bound=Callable)

_seen: set[tuple[str, str]] = set()
_counts: Counter[str] = Counter()
_summary_registered = False


def log_level() -> int:
    try:
        return int(os.environ.get("TENSCUDA_MIGRATION_LOG", "0"))
    except ValueError:
        return 0


def record(api: str, category: str, detail: str = "") -> None:
    """category: mapped | stub | unsupported"""
    lvl = log_level()
    if lvl <= 0:
        return
    key = (api, category)
    _counts[f"{category}:{api}"] += 1
    if lvl >= 2 and key not in _seen:
        _seen.add(key)
        msg = f"tenscuda migration [{category}] {api}"
        if detail:
            msg = f"{msg}: {detail}"
        log.warning(msg)


def wrap(api: str, category: str, fn: Callable, *, detail: str = "") -> Callable:
    """Call fn and log once per (api, category) when level >= 2."""

    def wrapped(*args, **kwargs):
        record(api, category, detail)
        return fn(*args, **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", api)
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


def status_for_cuda_name(name: str) -> str:
    entry = CUDA_API.get(name)
    if entry is None:
        return "stub"
    status = entry[0]
    if status == "Y":
        return "mapped"
    if status == "N":
        return "unsupported"
    return "stub"


def _print_summary() -> None:
    if log_level() < 1 or not _counts:
        return
    by_cat: Counter[str] = Counter()
    for key, n in _counts.items():
        cat, _ = key.split(":", 1)
        by_cat[cat] += n
    parts = [f"{k}={v}" for k, v in sorted(by_cat.items())]
    log.warning(
        "tenscuda migration summary: %s (%d unique apis touched)",
        ", ".join(parts),
        len(_counts),
    )


def register_summary_on_exit() -> None:
    global _summary_registered
    if _summary_registered or log_level() < 1:
        return
    atexit.register(_print_summary)
    _summary_registered = True


def reset_for_tests() -> None:
    _seen.clear()
    _counts.clear()