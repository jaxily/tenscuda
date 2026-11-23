"""Central env parsing for TensCUDA generate/bench (M4 compile + E4 native)."""
from __future__ import annotations

import os

_FALSE = frozenset({"0", "false", "no", "off", ""})


def compile_enabled(explicit: bool | None = None) -> bool:
    """Default compile unlock on (Nov 2026). Set ``TENSCUDA_COMPILE=0`` to force eager."""
    if explicit is not None:
        return explicit
    v = os.environ.get("TENSCUDA_COMPILE", "1").strip().lower()
    return v not in _FALSE


def native_ops_default_list() -> str:
    """Ops routed when native default is active (E4)."""
    return os.environ.get("TENSCUDA_NATIVE_OPS_LIST", "sdpa,rms_norm")


def native_default_active() -> bool:
    """Opt-in composite routing on compile path (enable after A/B parity)."""
    return os.environ.get("TENSCUDA_NATIVE_DEFAULT", "0").strip().lower() not in _FALSE


def ensure_native_ops_env() -> None:
    """Set ``TENSCUDA_NATIVE_OPS`` when compile + native default requested and unset."""
    if os.environ.get("TENSCUDA_NATIVE_OPS", "").strip():
        return
    if compile_enabled() and native_default_active():
        os.environ["TENSCUDA_NATIVE_OPS"] = native_ops_default_list()