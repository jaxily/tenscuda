"""Device-resident greedy decode; optional multi-step chunking.

Each decode forward must end with ``torch_xla.sync()`` before the next forward
when using ``StaticCache`` (see ``scripts/kv_diagnose.py``). Chunking only
amortizes host-side overhead between syncs; inner loop syncs per token.
"""
import os
import time

import torch
import torch_xla
from transformers import StaticCache

from .generate import _forward_decode_step, _forward_prefill


def generate(model, input_ids, max_new_tokens=32, eos_token_id=None, stats=None, chunk=1, pad_to=64):
    dev = next(model.parameters()).device
    ids_cpu = input_ids.detach().cpu()
    if ids_cpu.shape[0] != 1:
        raise NotImplementedError("resident generate supports batch size 1")
    plen = ids_cpu.shape[1]
    max_len = -(-(plen + max_new_tokens) // pad_to) * pad_to
    cache = StaticCache(config=model.config, max_cache_len=max_len)
    tokens = []
    timings = []
    with torch.no_grad():
        t = time.time()
        logits = _forward_prefill(model, model, ids_cpu, cache, dev)
        nxt = logits[:, -1, :].argmax(-1).reshape(1, 1)
        timings.append(time.time() - t)
        tokens.append(nxt)
        step_pos = plen
        n = 1
        while n < max_new_tokens:
            t = time.time()
            K = min(chunk, max_new_tokens - n)
            for _ in range(K):
                tok = int(nxt.reshape(-1).cpu())
                logits = _forward_decode_step(model, tok, step_pos, cache, dev)
                nxt = logits[:, -1, :].argmax(-1).reshape(1, 1)
                tokens.append(nxt)
                step_pos += 1
            timings.extend([(time.time() - t) / K] * K)
            n += K
    out = torch.cat(tokens, 1).cpu()
    if eos_token_id is not None:
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        for i, v in enumerate(out[0].tolist()):
            if v in eos:
                out = out[:, :i + 1]
                break
    if stats is not None:
        stats["step_seconds"] = timings
    return torch.cat([ids_cpu, out], 1).to(dev)