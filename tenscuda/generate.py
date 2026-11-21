"""Fixed-shape decode loop for Hugging Face causal LMs on TT.

HF's stock ``generate`` grows ``input_ids`` by one token per step and checks
stopping criteria on the device. Under torch-xla every new shape is a new
graph, so the stock loop triggers a fresh TT-MLIR compile for every token.

This loop keeps all shapes constant after the first step:
  * a ``StaticCache`` sized to ``prompt + max_new_tokens`` (rounded up),
  * one prefill graph (prompt length), and one 1-token decode graph reused
    for every later token,
  * token bookkeeping and stopping checks on the host CPU.
"""
from __future__ import annotations

import os
import time

import torch


def _sync(dev=None):
    import torch_xla

    torch_xla.sync()


def _compiled_forward(model):
    """torch.compile(backend="tt") once per model: dynamo captures the graph, so
    later steps skip re-tracing the Python model (the lazy-tensor cost)."""
    fn = getattr(model, "_tenscuda_compiled", None)
    if fn is None:
        from ._patch import real_torch_device

        compiled = torch.compile(model, backend=os.environ.get("TENSCUDA_COMPILE_BACKEND", "tt"), dynamic=False)

        def fn(*a, **k):
            with real_torch_device():
                return compiled(*a, **k)

        model._tenscuda_compiled = fn
    return fn


def static_greedy_generate(model, input_ids, max_new_tokens=32, eos_token_id=None,
                           attention_mask=None, pad_to=64, stats=None, compile=None):
    from transformers import StaticCache

    dev = next(model.parameters()).device
    ids_cpu = input_ids.detach().to("cpu")
    if ids_cpu.shape[0] != 1:
        raise NotImplementedError("tenscuda static generate: batch size 1 only")
    if attention_mask is not None and not bool((attention_mask.detach().to("cpu") == 1).all()):
        raise NotImplementedError("tenscuda static generate: padded attention_mask not supported")
    plen = ids_cpu.shape[1]
    max_len = -(-(plen + max_new_tokens) // pad_to) * pad_to
    cache = StaticCache(config=model.config, max_cache_len=max_len)
    eos = set()
    if eos_token_id is not None:
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])

    if compile is None:
        compile = os.environ.get("TENSCUDA_COMPILE", "0") == "1"
    fwd = _compiled_forward(model) if compile else model

    out = []
    timings = []
    with torch.no_grad():
        t = time.time()
        pos = torch.arange(plen).to(dev)
        logits = fwd(input_ids=ids_cpu.to(dev), cache_position=pos, past_key_values=cache,
                       use_cache=True).logits
        _sync()
        nxt = logits[:, -1, :].argmax(-1)
        tok = int(nxt.cpu())
        timings.append(time.time() - t)
        out.append(tok)
        cur = plen
        while len(out) < max_new_tokens and tok not in eos:
            t = time.time()
            step_ids = torch.tensor([[tok]]).to(dev)
            step_pos = torch.tensor([cur]).to(dev)
            logits = fwd(input_ids=step_ids, cache_position=step_pos, past_key_values=cache,
                           use_cache=True).logits
            _sync()
            tok = int(logits[:, -1, :].argmax(-1).cpu())
            timings.append(time.time() - t)
            out.append(tok)
            cur += 1
    if stats is not None:
        stats["step_seconds"] = timings
    return torch.cat([ids_cpu, torch.tensor([out])], dim=1).to(dev)
