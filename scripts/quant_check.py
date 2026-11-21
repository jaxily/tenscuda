"""Demonstrates the PITFALL: weight_dtype_override without a parameter mark is a no-op, and in-process graph reuse hides dtype changes. Use scripts/quant_logits.py."""
import sys, torch
import tenscuda
from tenscuda.loader import load_causal_lm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tt_torch.weight_dtype import apply_weight_dtype_overrides, remove_weight_dtype_overrides
mid = sys.argv[1]
tok = AutoTokenizer.from_pretrained(mid)
ids = tok.apply_chat_template([{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
with torch.no_grad():
    ref = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()(ids).logits[0].float()
    m = load_causal_lm(mid)
    def rel(x): return ((x - ref).norm() / ref.norm()).item()
    for w in (None, "bfp_bf8", "bfp_bf4"):
        if w:
            remove_weight_dtype_overrides(m)
            apply_weight_dtype_overrides(m, {"model.layers.*.weight": w})
        lg = m(ids.cuda()).logits[0].float().cpu()
        print("QUANT", w or "bf16", "logits_rel_err", round(rel(lg), 5), "argmax_agree", round((lg.argmax(-1) == ref.argmax(-1)).float().mean().item(), 4), flush=True)
