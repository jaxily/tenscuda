"""Teacher-forced logits error vs a CPU fp32 reference, one weight dtype per process.

    python scripts/quant_logits.py ref MODEL               # CPU fp32 -> results/ref_logits_<model>.pt
    python scripts/quant_logits.py tt MODEL bf16|bfp_bf8|bfp_bf4
Sequence = chat prompt + the CPU fp32 greedy continuation (so every position is scored).
"""
import json, sys, torch
mode, mid = sys.argv[1], sys.argv[2]
short = mid.split("/")[-1]
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(mid)
ids = tok.apply_chat_template([{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
if mode == "ref":
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    with torch.no_grad():
        seq = m.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        lg = m(seq).logits[0].float()
    torch.save({"seq": seq, "logits": lg}, f"results/ref_logits_{short}.pt")
    print("saved", seq.shape)
    sys.exit()
import tenscuda
from tenscuda.loader import load_causal_lm
ref = torch.load(f"results/ref_logits_{short}.pt")
m = load_causal_lm(mid)
w = sys.argv[3]
if w != "bf16":
    from tenscuda.quant import quantize_weights
    n = len(quantize_weights(m, w))
else:
    n = 0
with torch.no_grad():
    lg = m(ref["seq"].cuda()).logits[0].float().cpu()
r = ref["logits"]
p = ids.shape[1] - 1
res = {"model": mid, "weight_dtype": w, "n_quantized": n,
       "logits_rel_l2": ((lg - r).norm() / r.norm()).item(),
       "argmax_agree_all": (lg.argmax(-1) == r.argmax(-1)).float().mean().item(),
       "argmax_agree_generated": (lg[p:].argmax(-1) == r[p:].argmax(-1)).float().mean().item(),
       "kl_generated": torch.nn.functional.kl_div(lg[p:].log_softmax(-1), r[p:].log_softmax(-1), log_target=True, reduction="batchmean").item()}
print("QLOGITS", json.dumps(res), flush=True)
json.dump(res, open(f"results/quant_logits_{short}_{w}.json", "w"), indent=1)
