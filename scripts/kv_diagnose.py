#!/usr/bin/env python3
"""Compare StaticCache decode: per-step sync (safe) vs batched forwards (broken).

Run on a P150a with tenscuda installed, e.g.:

  python scripts/kv_diagnose.py HuggingFaceTB/SmolLM2-135M-Instruct 8

Writes JSON to results/kv_diagnose_<model_slug>.json and prints a short report.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def _prompt_ids(tok):
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    if tok.chat_template:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    return tok(msgs[0]["content"], return_tensors="pt").input_ids


def _decode_chunked_one_sync(model, ids, n_new, dev, chunk=4, pad_to=64):
    """Pre-fix resident pattern: K forwards per launch, one sync (corrupts StaticCache on TT)."""
    import torch
    import torch_xla
    from transformers import StaticCache

    plen = ids.shape[1]
    max_len = -(-(plen + n_new) // pad_to) * pad_to
    cache = StaticCache(config=model.config, max_cache_len=max_len)
    out = []
    with torch.no_grad():
        pos = torch.arange(plen, device=dev)
        logits = model(
            input_ids=ids.to(dev), cache_position=pos, past_key_values=cache, use_cache=True,
        ).logits
        nxt = logits[:, -1, :].argmax(-1).reshape(1, 1)
        torch_xla.sync()
        out.append(int(nxt.cpu()))
        step_pos = plen
        n = 1
        while n < n_new:
            K = min(chunk, n_new - n)
            for _ in range(K):
                step_p = torch.tensor([step_pos], device=dev)
                logits = model(
                    input_ids=nxt, cache_position=step_p, past_key_values=cache, use_cache=True,
                ).logits
                nxt = logits[:, -1, :].argmax(-1).reshape(1, 1)
                out.append(int(nxt.cpu()))
                step_pos += 1
            torch_xla.sync()
            n += K
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__.strip())
        sys.exit(2)
    model_id = sys.argv[1]
    n_new = int(sys.argv[2])
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tenscuda.generate import static_greedy_generate
    from tenscuda.resident import generate as resident_generate

    if not tenscuda.is_active:
        sys.exit("tenscuda: no TT device")

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = _prompt_ids(tok)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    dev = next(m.parameters()).device
    eos = tok.eos_token_id

    t0 = time.time()
    safe = static_greedy_generate(m, ids, max_new_tokens=n_new, eos_token_id=eos)
    safe_toks = safe[0, ids.shape[1]:].cpu().tolist()
    safe_s = time.time() - t0

    t0 = time.time()
    res = resident_generate(m, ids, max_new_tokens=n_new, eos_token_id=eos, chunk=chunk)
    res_toks = res[0, ids.shape[1]:].cpu().tolist()
    res_s = time.time() - t0

    t0 = time.time()
    batched = _decode_chunked_one_sync(m, ids, n_new, dev, chunk=chunk)
    batched_s = time.time() - t0

    agree_res = sum(a == b for a, b in zip(safe_toks, res_toks))
    agree_bat = sum(a == b for a, b in zip(safe_toks, batched))

    report = {
        "model": model_id,
        "n_new": n_new,
        "chunk": chunk,
        "safe_tokens": safe_toks,
        "resident_sync_per_step_tokens": res_toks,
        "chunked_one_sync_per_launch_tokens": batched,
        "parity_resident_vs_safe": f"{agree_res}/{max(len(safe_toks), len(res_toks))}",
        "parity_batched_vs_safe": f"{agree_bat}/{max(len(safe_toks), len(batched))}",
        "seconds": {"safe": round(safe_s, 2), "resident": round(res_s, 2), "batched_bug": round(batched_s, 2)},
        "diagnosis": (
            "StaticCache must see one torch_xla.sync() between each decode forward. "
            "Resident chunking with per-step sync should match static_greedy_generate; "
            "deferring sync across multiple forwards in one lazy graph yields wrong tokens."
        ),
    }
    out_path = ROOT / "results" / f"kv_diagnose_{slug(model_id)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "safe_tokens"}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()