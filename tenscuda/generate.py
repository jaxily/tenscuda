"""Fixed-shape decode loop for Hugging Face causal LMs on TT.

HF's stock ``generate`` grows ``input_ids`` by one token per step and checks
stopping criteria on the device. Under torch-xla every new shape is a new
graph, so the stock loop triggers a fresh TT-MLIR compile for every token.

Neuron-style pattern (transformers-neuronx / NxD inference):
  * separate prefill vs single-token decode launches,
  * one XLA graph boundary per token (``_sync()`` after each forward),
  * ``StaticCache`` updated between launches, not batched inside one lazy graph.

Batching K>1 decode steps without a sync between steps corrupts ``StaticCache``
(see ``scripts/kv_diagnose.py`` and docs/PERF.md).
"""
from __future__ import annotations

import os
import time

import torch


def _sync(dev=None):
    import torch_xla

    torch_xla.sync()


def _xla_compile_count() -> int:
    import torch_xla.debug.metrics as met

    row = met.metric_data("CompileTime")
    return int(row[0]) if row else 0


class _ResidentDecodeIO:
    """Fixed on-device shapes for 1-token decode (M4: fewer host tensor factories)."""

    def __init__(self, dev):
        self.step_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.step_pos = torch.zeros(1, dtype=torch.long, device=dev)

    def write(self, token_id: int, cache_position: int):
        self.step_ids[0, 0] = token_id
        self.step_pos[0] = cache_position


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


def _forward_prefill(fwd, model, input_ids, cache, dev):
    plen = input_ids.shape[1]
    pos = torch.arange(plen, device=dev)
    out = fwd(
        input_ids=input_ids.to(dev),
        cache_position=pos,
        past_key_values=cache,
        use_cache=True,
    )
    _sync()
    return out.logits


def _forward_decode_step(
    fwd,
    token_id: int,
    cache_position: int,
    cache,
    dev,
    *,
    decode_io: _ResidentDecodeIO | None = None,
    stats: dict | None = None,
):
    """One decode token; must end with ``_sync()`` before the next step."""
    comp_before = _xla_compile_count() if stats is not None else None
    if decode_io is not None:
        decode_io.write(token_id, cache_position)
        step_ids, step_pos = decode_io.step_ids, decode_io.step_pos
    else:
        step_ids = torch.tensor([[token_id]], device=dev)
        step_pos = torch.tensor([cache_position], device=dev)
    out = fwd(
        input_ids=step_ids,
        cache_position=step_pos,
        past_key_values=cache,
        use_cache=True,
    )
    _sync()
    if stats is not None and comp_before is not None:
        after = _xla_compile_count()
        stats.setdefault("xla_compiles_per_step", []).append(after)
        stats.setdefault("xla_compile_deltas", []).append(after - comp_before)
    return out.logits


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
    decode_io = _ResidentDecodeIO(dev) if os.environ.get("TENSCUDA_RESIDENT_DECODE_IO", "0") == "1" else None
    if stats is None and os.environ.get("TENSCUDA_M4_DEBUG", "0") == "1":
        stats = {}

    out = []
    timings = []
    with torch.no_grad():
        t = time.time()
        logits = _forward_prefill(fwd, model, ids_cpu, cache, dev)
        nxt = int(logits[:, -1, :].argmax(-1).cpu())
        timings.append(time.time() - t)
        out.append(nxt)
        cur = plen
        while len(out) < max_new_tokens and nxt not in eos:
            t = time.time()
            logits = _forward_decode_step(
                fwd, nxt, cur, cache, dev, decode_io=decode_io, stats=stats,
            )
            nxt = int(logits[:, -1, :].argmax(-1).cpu())
            timings.append(time.time() - t)
            out.append(nxt)
            cur += 1
    if stats is not None:
        stats["step_seconds"] = timings
    return torch.cat([ids_cpu, torch.tensor([out])], dim=1).to(dev)