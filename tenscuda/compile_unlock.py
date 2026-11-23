"""M4 compile unlock: torch.compile decode with modes exited (Nov 23 overnight).

Do not casually refactor this loop — an inlined twin in generate.py failed A/B
(2/16) while this exact body matched 32/32. Call ``compile_decode_steps``.
"""
from __future__ import annotations

import os
import time
from typing import Any


def compile_decode_steps(model: Any, ids_cpu, n_new: int, *, use_compile: bool = True) -> list[dict]:
    """Run prefill+decode; return per-step token/logit/ms records.

    When ``use_compile`` is True: CPU-init StaticCache, exit torch-function modes,
    ``torch.compile(backend="tt")``, float top-1, per-step sync.
    """
    import torch
    from transformers import StaticCache

    from tenscuda.generate import _compiled_forward, _sync

    os.environ["TENSCUDA_STATIC_CACHE_CPU_INIT"] = "1"
    if use_compile:
        os.environ["TENSCUDA_COMPILE"] = "1"
        os.environ["TENSCUDA_DECODE_PAUSE_MODES"] = "1"
    else:
        os.environ["TENSCUDA_COMPILE"] = "0"
        os.environ["TENSCUDA_DECODE_PAUSE_MODES"] = "0"

    dev = next(model.parameters()).device
    plen = ids_cpu.shape[1]
    max_len = -(-(plen + n_new) // 64) * 64
    cache = StaticCache(
        config=model.config, max_cache_len=max_len, device="cpu",
        dtype=torch.bfloat16, max_batch_size=1,
    )
    cfg = model.config
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    cache.early_initialization(
        batch_size=1, num_heads=n_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu",
    )
    for layer in cache.layers:
        layer.keys = layer.keys.to(dev)
        layer.values = layer.values.to(dev)
        if hasattr(layer.cumulative_length, "to"):
            layer.cumulative_length = layer.cumulative_length.to(dev)
        layer.device = dev

    exited: list = []
    if use_compile:
        import tenscuda._patch as patch

        mode = patch._S.originals.get("mode")
        if mode is not None:
            mode.__exit__(None, None, None)
            exited.append(mode)
        try:
            import tt_torch.torch_overrides as overrides

            tt = overrides.torch_function_override
            tt.__exit__(None, None, None)
            exited.append(tt)
        except Exception:
            pass
        if hasattr(model, "_tenscuda_compiled"):
            delattr(model, "_tenscuda_compiled")
        fwd = _compiled_forward(model)
    else:
        fwd = model

    steps: list[dict] = []
    try:
        with torch.no_grad():
            pos = torch.arange(plen, device=dev)
            t0 = time.time()
            logits = fwd(
                input_ids=ids_cpu.to(dev), cache_position=pos, past_key_values=cache, use_cache=True,
            ).logits
            _sync()
            last = logits[:, -1, :].float().cpu()
            top2 = torch.topk(last, 2, dim=-1)
            tok = int(top2.indices[0, 0])
            steps.append({
                "i": 0, "phase": "prefill", "token": tok,
                "top1": tok, "top1_logit": float(top2.values[0, 0]),
                "top2": int(top2.indices[0, 1]), "top2_logit": float(top2.values[0, 1]),
                "margin": float(top2.values[0, 0] - top2.values[0, 1]),
                "ms": round((time.time() - t0) * 1000, 1),
            })
            cur = plen
            nxt = tok
            for i in range(1, n_new):
                t0 = time.time()
                logits = fwd(
                    input_ids=torch.tensor([[nxt]], device=dev),
                    cache_position=torch.tensor([cur], device=dev),
                    past_key_values=cache,
                    use_cache=True,
                ).logits
                _sync()
                last = logits[:, -1, :].float().cpu()
                top2 = torch.topk(last, 2, dim=-1)
                tok = int(top2.indices[0, 0])
                steps.append({
                    "i": i, "phase": "decode", "token": tok,
                    "top1": tok, "top1_logit": float(top2.values[0, 0]),
                    "top2": int(top2.indices[0, 1]), "top2_logit": float(top2.values[0, 1]),
                    "margin": float(top2.values[0, 0] - top2.values[0, 1]),
                    "ms": round((time.time() - t0) * 1000, 1),
                    "cache_pos": cur,
                })
                nxt = tok
                cur += 1
    finally:
        for ctx in reversed(exited):
            try:
                ctx.__enter__()
            except Exception:
                pass
    return steps
