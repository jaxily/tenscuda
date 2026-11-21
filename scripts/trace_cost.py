"""Host cost per decode step split into trace (Python->IR) vs sync (hash+execute), per config.

    python scripts/trace_cost.py MODEL CONFIG   CONFIG in: default | no_tt_override | no_modes
"""
import sys, time, json, torch
import tenscuda
import torch_xla
from tenscuda.loader import load_causal_lm
from transformers import StaticCache, AutoTokenizer
mid, cfg = sys.argv[1], sys.argv[2]
if cfg in ("no_tt_override", "no_modes"):
    import tt_torch.torch_overrides as o
    o.torch_function_override.__exit__(None, None, None)
if cfg == "no_modes":
    tenscuda._patch._S.originals["mode"].__exit__(None, None, None)
m = load_causal_lm(mid, verbose=False)
tok = AutoTokenizer.from_pretrained(mid)
ids = tok.apply_chat_template([{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
dev = tenscuda.device
cache = StaticCache(config=m.config, max_cache_len=128)
with torch.no_grad():
    lg = m(input_ids=ids.to(dev), cache_position=torch.arange(ids.shape[1]).to(dev), past_key_values=cache).logits
    t = int(lg[0, -1].argmax().cpu()); toks = [t]; pos = ids.shape[1]
    tr, sy = [], []
    for i in range(40):
        a = time.time()
        lg = m(input_ids=torch.tensor([[t]]).to(dev), cache_position=torch.tensor([pos]).to(dev), past_key_values=cache).logits
        nxt = lg[0, -1].argmax()
        b = time.time(); torch_xla.sync(); t = int(nxt.cpu()); c = time.time()
        toks.append(t); pos += 1
        if i >= 8: tr.append(b - a); sy.append(c - b)
print("TRACE", json.dumps({"config": cfg, "trace_ms": round(1000 * sum(tr) / len(tr), 1), "sync_exec_ms": round(1000 * sum(sy) / len(sy), 1), "tokens": toks[:32]}), flush=True)
