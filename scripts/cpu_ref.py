"""CPU greedy reference for the compatibility matrix.

    python scripts/cpu_ref.py MODEL DTYPE NEW_TOKENS OUT.json

Uses a plain greedy loop (no generation_config: no repetition penalty, no EOS stop), the
same as tenscuda.generate.static_greedy_generate, so token streams are comparable.
Saves per-step top1-top2 logit margins, so a TT divergence can be checked against
a near-tie on the CPU.
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import prompt_ids  # noqa: E402  (scripts/prompt.py)

mid, dtype, n, out = sys.argv[1], getattr(torch, sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.set_num_threads(4)
import threading
_peak = {"anon": 0.0}


def _anon():
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def _watch():
    while True:
        _peak["anon"] = max(_peak["anon"], _anon()); time.sleep(0.05)


threading.Thread(target=_watch, daemon=True).start()
tok = AutoTokenizer.from_pretrained(mid)
t = time.time()
m = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype, low_cpu_mem_usage=True).eval()
load_s = time.time() - t
ids = prompt_ids(tok)
toks, margins, steps = [], [], []
with torch.no_grad():
    t = time.time()
    o = m(input_ids=ids, use_cache=True)
    pkv = o.past_key_values
    for i in range(n):
        lg = o.logits[0, -1].float()
        top = lg.topk(2)
        nxt = int(top.indices[0])
        toks.append(nxt)
        margins.append(round(float(top.values[0] - top.values[1]), 4))
        steps.append(time.time() - t)
        t = time.time()
        if i + 1 < n:
            o = m(input_ids=torch.tensor([[nxt]]), past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values


def vmhwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2**20


dec = steps[1:]
json.dump({"model": mid, "dtype": sys.argv[2], "prompt_tokens": ids.shape[1], "load_s": round(load_s, 1),
           "decode_tok_per_s": round(len(dec) / sum(dec), 2) if dec else None,
           "peak_rss_gib": round(vmhwm(), 2), "peak_anon_gib": round(_peak["anon"], 2), "tokens": toks, "margins": margins,
           "text": tok.decode(toks, skip_special_tokens=True)}, open(out, "w"), indent=1)
print(json.dumps({"dtype": sys.argv[2], "decode_tok_per_s": round(len(dec) / sum(dec), 2), "peak_rss_gib": round(vmhwm(), 2)}))
