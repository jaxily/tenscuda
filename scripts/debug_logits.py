"""Compare TT vs CPU logits for one forward pass (no cache, then static cache)."""
import sys, torch
import tenscuda
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
mid = sys.argv[1]
tok = AutoTokenizer.from_pretrained(mid)
msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
cpu = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).eval()
with torch.no_grad():
    ref = cpu(ids).logits[0].float()
    tt = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).eval().cuda()
    def report(name, got):
        got = got[0].float().cpu()
        agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"{name}: maxdiff={ (got-ref).abs().max().item():.3f} argmax_agree={agree:.3f} last_tt={got[-1].argmax().item()} last_cpu={ref[-1].argmax().item()}", flush=True)
    report("plain", tt(ids.cuda()).logits)
    cache = StaticCache(config=tt.config, max_cache_len=128)
    report("static", tt(input_ids=ids.cuda(), cache_position=torch.arange(ids.shape[1]).cuda(), past_key_values=cache, use_cache=True).logits)
