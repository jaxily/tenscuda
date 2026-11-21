"""Accuracy overrides for ops whose fused TT kernels are much less accurate.

Measured on Blackhole P150a, pjrt-plugin-tt 1.5.0.dev20260831, fp32 inputs
(scripts/softmax_probe.py, scripts/div_probe.py):

  op                           rel L2 error vs CPU fp32
  F.softmax (fused ttnn)       2.26e-2
  exp(x-max) / sum             2.26e-2   (pattern-matched into the same fused op)
  exp(x-max) * (1/sum)         2.76e-4   <- what we emit instead
  F.scaled_dot_product_attn    2.44e-2   (uses the fused softmax)

Disable with TENSCUDA_EXACT_SOFTMAX=0.
"""
from __future__ import annotations

import math

import torch


def softmax(x, dim, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    e = torch.exp(x - x.amax(dim, keepdim=True))
    return e * torch.reciprocal(e.sum(dim, keepdim=True))


def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
         enable_gqa=False):
    L, S = query.shape[-2], key.shape[-2]
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    if enable_gqa and key.shape[-3] != query.shape[-3]:
        rep = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(rep, dim=-3)
        value = value.repeat_interleave(rep, dim=-3)
    scores = (query @ key.transpose(-2, -1)) * scale
    if is_causal:
        causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=S - L)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    p = softmax(scores.float(), -1).to(query.dtype)
    return p @ value


def _is_xla(t):
    return isinstance(t, torch.Tensor) and t.device.type == "xla"


SOFTMAX_FUNCS = {torch.softmax, torch.nn.functional.softmax, torch.Tensor.softmax,
                 torch.special.softmax}
SDPA_FUNCS = {torch.nn.functional.scaled_dot_product_attention}


def maybe_override(func, args, kwargs):
    """Return (handled, result)."""
    if func in SOFTMAX_FUNCS and args and _is_xla(args[0]):
        x = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
        if dim is None:
            return False, None
        return True, softmax(x, dim, dtype)
    if func in SDPA_FUNCS and args and _is_xla(args[0]):
        kw = dict(kwargs)
        if kw.get("dropout_p", 0.0) or (len(args) > 4 and args[4]):
            return False, None
        names = ["query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"]
        for n, a in zip(names, args):
            kw[n] = a
        return True, sdpa(**kw)
    return False, None
