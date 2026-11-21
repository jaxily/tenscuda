"""Per-layer hidden-state + logits comparison at the first divergent decode step.

Compares three runs on the same token prefix:
  ref  = CPU float32 (the "truth")
  cpub = CPU bfloat16
  ttb  = TT  bfloat16 (via tenscuda)
If TT-bf16's error vs fp32 is the same size as CPU-bf16's own error, the token
flip is bf16 rounding, not an op bug.

    python scripts/layer_diff.py MODEL CPU_BF16_JSON TT_BF16_JSON
"""
import json
import sys

import torch

import tenscuda  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

mid, cpu_json, tt_json = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(mid)
ids = tok.apply_chat_template([{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
a = json.load(open(cpu_json))["runs"][-1]["tokens"]
b = json.load(open(tt_json))["runs"][-1]["tokens"]
k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
if k is None:
    sys.exit("no divergence")
prefix = torch.cat([ids, torch.tensor([a[:k]])], 1)  # shared prefix right before the flip
print(f"first divergence at new-token index {k}: cpu={a[k]!r} {tok.decode([a[k]])!r} tt={b[k]!r} {tok.decode([b[k]])!r}")


def run(dtype, dev):
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype).eval()
    if dev == "cuda":
        m = m.cuda()
    with torch.no_grad():
        o = m(prefix.to(dev) if dev == "cuda" else prefix, output_hidden_states=True)
    hs = [h[0].float().cpu() for h in o.hidden_states]
    lg = o.logits[0, -1].float().cpu()
    del m
    return hs, lg


ref_h, ref_l = run(torch.float32, "cpu")
cpu_h, cpu_l = run(torch.bfloat16, "cpu")
tt_h, tt_l = run(torch.bfloat16, "cuda")


def rel(x, y):
    return ((x - y).norm() / y.norm()).item()


rows = []
for i, (r, c, t) in enumerate(zip(ref_h, cpu_h, tt_h)):
    rows.append({"layer": i, "cpu_bf16_vs_fp32": rel(c, r), "tt_bf16_vs_fp32": rel(t, r), "tt_vs_cpu_bf16": rel(t, c)})
for row in rows:
    print("layer {layer:2d}: cpu_bf16 {cpu_bf16_vs_fp32:.4f}  tt_bf16 {tt_bf16_vs_fp32:.4f}  tt_vs_cpu {tt_vs_cpu_bf16:.4f}".format(**row))


def top2(lg):
    v, i = lg.topk(2)
    return {"ids": i.tolist(), "tokens": [tok.decode([j]) for j in i.tolist()], "logits": v.tolist(),
            "margin": (v[0] - v[1]).item()}


res = {"model": mid, "divergence_index": k, "layers": rows,
       "logits_rel_err": {"cpu_bf16": rel(cpu_l, ref_l), "tt_bf16": rel(tt_l, ref_l)},
       "top2": {"cpu_fp32": top2(ref_l), "cpu_bf16": top2(cpu_l), "tt_bf16": top2(tt_l)}}
print(json.dumps(res["logits_rel_err"]), json.dumps(res["top2"]))
json.dump(res, open(f"results/layer_diff_{mid.split('/')[-1]}.json", "w"), indent=1)
