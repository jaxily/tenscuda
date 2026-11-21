"""Speed of the tenscuda static greedy loop on TT, per mode, N repeats.

    python scripts/speed.py MODEL DTYPE MODE(lazy|compile) NEW_TOKENS REPEATS
Writes results/speed_<model>_<dtype>_<mode>.json
"""
import json, sys, time
import torch
import tenscuda
from tenscuda.generate import static_greedy_generate
import torch_xla.debug.metrics as met
from transformers import AutoModelForCausalLM, AutoTokenizer

mid, dt, mode, n, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
tok = AutoTokenizer.from_pretrained(mid)
ids = tok.apply_chat_template([{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
m = AutoModelForCausalLM.from_pretrained(mid, dtype=getattr(torch, dt)).cuda().eval()
runs = []
for r in range(reps + 1):
    st = {}
    t = time.time()
    if mode.startswith("resident"):
        from tenscuda.resident import generate
        chunk = int(mode.split(":")[1]) if ":" in mode else 1
        o = generate(m, ids, max_new_tokens=n, eos_token_id=None, stats=st, chunk=chunk)
    else:
        o = static_greedy_generate(m, ids, max_new_tokens=n, eos_token_id=None, stats=st, compile=(mode == "compile"))
    dt_s = time.time() - t
    steps = st["step_seconds"]
    toks = o[0, ids.shape[1]:].cpu().tolist()
    dec = steps[1:]
    runs.append({"run": r, "total_s": round(dt_s, 3), "prefill_s": round(steps[0], 3),
                 "decode_tok_per_s": round(len(dec) / sum(dec), 2), "e2e_tok_per_s": round(len(toks) / dt_s, 2),
                 "first_decode_step_s": round(dec[0], 3), "tokens": toks})
    print(json.dumps({k: v for k, v in runs[-1].items() if k != "tokens"}), flush=True)
cm = met.metric_data("CompileTime")
res = {"model": mid, "dtype": dt, "mode": mode, "new_tokens": n, "prompt_tokens": ids.shape[1],
       "xla_compiles": cm[0] if cm else 0, "xla_compile_s": round(cm[1] / 1e9, 1) if cm else 0,
       "cpu_fallback_ops": sorted(met.executed_fallback_ops()), "runs": runs}
print("compiles", res["xla_compiles"], "compile_s", res["xla_compile_s"], "fallback", res["cpu_fallback_ops"])
json.dump(res, open(f"results/speed_{mid.split('/')[-1]}_{dt}_{mode}.json", "w"), indent=1)
