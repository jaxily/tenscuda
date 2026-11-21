"""Device-resident greedy decode with K steps per graph / host launch.

No per-token CPU tensor reads. Tokens and positions stay on the TT device;
only the completed output is read back to the host. An EOS token truncates the
returned output, but computation runs to the requested max length.
"""
import os
import time

import torch
import torch_xla
from transformers import StaticCache


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
        pos = torch.arange(plen).to(dev)
        logits = model(input_ids=ids_cpu.to(dev), cache_position=pos, past_key_values=cache, use_cache=True).logits
        nxt = logits[:, -1, :].argmax(-1).reshape(1, 1)
        torch_xla.sync(wait=True)
        timings.append(time.time() - t)
        tokens.append(nxt)
        step_pos = torch.tensor([plen]).to(dev)
        n = 1
        while n < max_new_tokens:
            t = time.time()
            K = min(chunk, max_new_tokens - n)
            for _ in range(K):
                logits = model(input_ids=nxt, cache_position=step_pos, past_key_values=cache, use_cache=True).logits
                nxt = logits[:, -1, :].argmax(-1).reshape(1, 1)
                tokens.append(nxt)
                step_pos = step_pos + 1
            torch_xla.sync(wait=True)
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
