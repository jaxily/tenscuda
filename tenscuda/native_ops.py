"""E4: route hot ops through tt_torch StableHLO composites (native ttnn lowering).

Opt-in via ``TENSCUDA_NATIVE_OPS`` (comma list: ``sdpa``, ``rms_norm``, or ``all``), or
``TENSCUDA_NATIVE_DEFAULT=1`` to apply ``sdpa,rms_norm`` on import when unset (off by default:
A/B Nov 23 — 32/32 parity, no meaningful decode win on 8B).
Uses sanctioned ``tt_torch.composite_ops`` — no hand-written metal in this repo.
"""
from __future__ import annotations

import os

import torch

_RMS_FUNCS = {torch.rms_norm, torch.nn.functional.rms_norm}
SDPA_FUNCS = {torch.nn.functional.scaled_dot_product_attention}


def _enabled(kind: str) -> bool:
    raw = os.environ.get("TENSCUDA_NATIVE_OPS", "").strip().lower()
    if not raw:
        return False
    if raw in ("1", "all", "true", "yes"):
        return True
    return kind in {p.strip() for p in raw.split(",")}


def _is_xla(t):
    return isinstance(t, torch.Tensor) and t.device.type == "xla"


def maybe_override(func, args, kwargs):
    """Return (handled, result)."""
    if not args or not _is_xla(args[0]):
        return False, None
    try:
        from tt_torch import composite_ops as co
    except ImportError:
        return False, None

    if func in _RMS_FUNCS and _enabled("rms_norm"):
        inp = args[0]
        if len(args) > 1:
            norm_shape = args[1]
        else:
            norm_shape = kwargs.get("normalized_shape")
        weight = args[2] if len(args) > 2 else kwargs.get("weight")
        eps = args[3] if len(args) > 3 else kwargs.get("eps", 1e-6)
        if norm_shape is None:
            return False, None
        return True, co.composite_rms_norm(inp, norm_shape, weight, eps)

    if func in SDPA_FUNCS and _enabled("sdpa"):
        if kwargs.get("dropout_p", 0.0) or (len(args) > 4 and args[4]):
            return False, None
        names = [
            "query", "key", "value", "attn_mask", "dropout_p", "is_causal",
            "scale", "enable_gqa",
        ]
        kw = dict(kwargs)
        for n, a in zip(names, args):
            kw[n] = a
        return True, co.composite_scaled_dot_product_attention(
            kw["query"], kw["key"], kw["value"],
            attn_mask=kw.get("attn_mask"),
            dropout_p=kw.get("dropout_p", 0.0),
            is_causal=kw.get("is_causal", False),
            scale=kw.get("scale"),
            enable_gqa=kw.get("enable_gqa", False),
        )

    return False, None