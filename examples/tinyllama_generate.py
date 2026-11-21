"""A plain CUDA-style Hugging Face script. It contains no TensCUDA code.

    python examples/tinyllama_generate.py               # CPU (no CUDA here)
    python -m tenscuda examples/tinyllama_generate.py   # Tenstorrent P150
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
p.add_argument("--max-new-tokens", type=int, default=32)
p.add_argument("--runs", type=int, default=2, help="run 1 includes compile/warm-up")
p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
p.add_argument("--out", default=None)
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

tok = AutoTokenizer.from_pretrained(a.model)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(device).eval()
load_s = time.time() - t0

msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
if tok.chat_template:
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)["input_ids"]
else:
    ids = tok(msgs[0]["content"], return_tensors="pt").input_ids
ids = ids.to(device)

runs = []
with torch.no_grad():
    for r in range(a.runs):
        t = time.time()
        out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        new = out[0, ids.shape[1]:].cpu().tolist()
        dt = time.time() - t
        runs.append({"seconds": round(dt, 3), "new_tokens": len(new),
                     "tok_per_s": round(len(new) / dt, 3), "tokens": new})
        print(f"run {r}: {len(new)} tokens in {dt:.2f}s = {len(new)/dt:.2f} tok/s")

print("text:", tok.decode(runs[-1]["tokens"], skip_special_tokens=True))
res = {"device": device, "dtype": a.dtype, "model": a.model, "prompt_tokens": ids.shape[1],
       "load_seconds": round(load_s, 2), "runs": runs}
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
