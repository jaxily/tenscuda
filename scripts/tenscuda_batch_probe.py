#!/usr/bin/env python3
"""Experimental batched greedy decode on TT-XLA (not production generate).

Measures per-user and aggregate tok/s for identical prompts × batch size.
KV: one sync per token step (same contract as static_greedy_generate).

Example (card host):
  PYTHONPATH=. .venv/bin/python scripts/tenscuda_batch_probe.py \\
    HuggingFaceTB/SmolLM2-135M-Instruct --batches 1,2,4 --tokens 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sync():
    import torch_xla

    torch_xla.sync()


def _make_cache(model, max_len, dev, batch_size: int):
    from transformers import StaticCache

    try:
        cache = StaticCache(
            config=model.config,
            max_cache_len=max_len,
            device="cpu",
            dtype=torch.bfloat16,
            max_batch_size=batch_size,
        )
    except TypeError:
        cache = StaticCache(config=model.config, max_cache_len=max_len)
    if hasattr(cache, "early_initialization"):
        cfg = model.config
        n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        cache.early_initialization(
            batch_size=batch_size,
            num_heads=n_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys.to(dev)
        if getattr(layer, "values", None) is not None:
            layer.values = layer.values.to(dev)
        cl = getattr(layer, "cumulative_length", None)
        if cl is not None and hasattr(cl, "to"):
            layer.cumulative_length = cl.to(dev)
        if hasattr(layer, "device"):
            layer.device = dev
    return cache


def batch_greedy_generate(model, input_ids, max_new_tokens: int, *, pad_to: int = 64):
    from tenscuda.decode_graph import get_forward_fn, pause_decode_trace_modes

    ids_cpu = input_ids.detach().to("cpu")
    bsz, plen = ids_cpu.shape
    dev = next(model.parameters()).device
    max_len = -(-(plen + max_new_tokens) // pad_to) * pad_to
    cache = _make_cache(model, max_len, dev, bsz)
    fwd = get_forward_fn(model)
    eos = getattr(model.config, "eos_token_id", None)
    eos_set = set()
    if eos is not None:
        eos_set = {eos} if not isinstance(eos, (list, tuple)) else set(eos)

    done = torch.zeros(bsz, dtype=torch.bool)
    tokens = [[] for _ in range(bsz)]
    timings: list[float] = []

    with torch.no_grad():
        t0 = time.time()
        pos = torch.arange(plen, device=dev)
        out = fwd(
            input_ids=ids_cpu.to(dev),
            cache_position=pos,
            past_key_values=cache,
            use_cache=True,
        )
        _sync()
        timings.append(time.time() - t0)
        nxt = out.logits[:, -1, :].argmax(-1).detach().cpu()
        for i in range(bsz):
            tid = int(nxt[i])
            tokens[i].append(tid)
            if tid in eos_set:
                done[i] = True

        cur = plen
        while max(len(x) for x in tokens) < max_new_tokens and not done.all():
            t0 = time.time()
            step_ids = nxt.to(dev).view(bsz, 1)
            step_pos = torch.full((bsz,), cur, dtype=torch.long, device=dev)
            with pause_decode_trace_modes():
                out = fwd(
                    input_ids=step_ids,
                    cache_position=step_pos,
                    past_key_values=cache,
                    use_cache=True,
                )
            _sync()
            timings.append(time.time() - t0)
            nxt = out.logits[:, -1, :].argmax(-1).detach().cpu()
            for i in range(bsz):
                if done[i]:
                    continue
                tid = int(nxt[i])
                tokens[i].append(tid)
                if tid in eos_set or len(tokens[i]) >= max_new_tokens:
                    done[i] = True
            cur += 1

    decode_s = sum(timings[1:])
    n_tok = sum(len(t) for t in tokens)
    return {
        "batch_size": bsz,
        "decode_tokens_total": n_tok,
        "decode_seconds": decode_s,
        "aggregate_tok_s": round(n_tok / decode_s, 2) if decode_s > 0 else None,
        "per_user_tok_s": round((n_tok / bsz) / decode_s, 2) if decode_s > 0 else None,
        "ms_per_step": round(1000.0 * decode_s / max(len(timings) - 1, 1), 2),
    }


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--batches", default="1,2,4", help="comma-separated batch sizes")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--out", type=Path, help="Write JSON here")
    args = ap.parse_args()

    import tenscuda  # noqa: F401

    os.environ.setdefault("TENSCUDA_COMPILE", "0")
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    if tok.chat_template:
        row = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    else:
        row = tok(msgs[0]["content"], return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    rows = []
    for b in batches:
        ids = row.repeat(b, 1)
        # warm
        try:
            batch_greedy_generate(model, ids, max_new_tokens=4)
        except Exception as e:
            rows.append({"batch_size": b, "error": str(e)})
            continue
        stats = batch_greedy_generate(model, ids, max_new_tokens=args.tokens)
        rows.append(stats)
        print(f"batch={b}: aggregate {stats.get('aggregate_tok_s')} tok/s, "
              f"per-user {stats.get('per_user_tok_s')} tok/s, "
              f"{stats.get('ms_per_step')} ms/step")

    doc = {"model": args.model, "tokens": args.tokens, "runs": rows}
    out = args.out or ROOT / "results" / f"tenscuda_batch_probe_{args.model.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()