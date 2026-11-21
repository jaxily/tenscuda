"""Peak host RAM + parity for loading a model onto TT.

    python scripts/load_bench.py MODEL MODE(stream|hf) NEW_TOKENS [WEIGHT_DTYPE]
MODE stream = tenscuda.loader (meta skeleton + per-tensor upload)
MODE hf     = from_pretrained(...).cuda()  (stages the whole model in host RAM)
WEIGHT_DTYPE = bfp_bf8 | bfp_bf4 (tt_torch weight_dtype_override on decoder Linear weights)
Writes results/load_<model>_<mode>[_<wdt>].json
"""
import json, sys, time
import torch
import tenscuda
from tenscuda.loader import load_causal_lm, peak_rss_gib
from tenscuda.generate import static_greedy_generate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch_xla.debug.metrics as met

mid, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
wdt = sys.argv[4] if len(sys.argv) > 4 else None
base = peak_rss_gib()
t = time.time()
if mode == "stream":
    m = load_causal_lm(mid, dtype=torch.bfloat16)
else:
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
load_s = time.time() - t
rss_after_load = peak_rss_gib()
applied = []
if wdt:
    from tenscuda.quant import quantize_weights
    applied = quantize_weights(m, wdt)
tok = AutoTokenizer.from_pretrained(mid)
msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"] \
    if tok.chat_template else tok(msgs[0]["content"], return_tensors="pt").input_ids
runs = []
for r in range(4):
    st = {}
    t = time.time()
    o = static_greedy_generate(m, ids, max_new_tokens=n, eos_token_id=None, stats=st)
    total = time.time() - t
    dec = st["step_seconds"][1:]
    runs.append({"run": r, "total_s": round(total, 2), "prefill_s": round(st["step_seconds"][0], 3),
                 "decode_tok_per_s": round(len(dec) / sum(dec), 2), "tokens": o[0, ids.shape[1]:].cpu().tolist()})
    print(json.dumps({k: v for k, v in runs[-1].items() if k != "tokens"}), flush=True)
cm = met.metric_data("CompileTime")
res = {"model": mid, "mode": mode, "weight_dtype": wdt, "n_overridden": len(applied),
       "load_s": round(load_s, 1), "peak_rss_gib_baseline": round(base, 2),
       "peak_rss_gib_after_load": round(rss_after_load, 2), "peak_rss_gib_end": round(peak_rss_gib(), 2),
       "xla_compiles": cm[0] if cm else 0, "xla_compile_s": round(cm[1] / 1e9, 1) if cm else 0,
       "cpu_fallback_ops": sorted(met.executed_fallback_ops()), "runs": runs,
       "text": tok.decode(runs[-1]["tokens"], skip_special_tokens=True)}
print(json.dumps({k: v for k, v in res.items() if k != "runs"}), flush=True)
tag = f"_{wdt}" if wdt else ""
json.dump(res, open(f"results/load_{mid.split('/')[-1]}_{mode}{tag}.json", "w"), indent=1)
