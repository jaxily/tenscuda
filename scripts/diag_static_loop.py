import sys, time, json, torch
import tenscuda
from tenscuda.generate import static_greedy_generate
import torch_xla.debug.metrics as met
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = sys.argv[1]; n = int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
msgs=[{"role":"user","content":"Explain in two sentences why the sky is blue."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"] if tok.chat_template else tok(msgs[0]["content"], return_tensors="pt").input_ids
for r in range(2):
    st = {}
    t=time.time(); o = static_greedy_generate(m, ids, max_new_tokens=n, eos_token_id=tok.eos_token_id, stats=st); dt=time.time()-t
    print("run", r, "total", round(dt,2), "steps", [round(x,2) for x in st["step_seconds"]], flush=True)
    print("CompileTime", met.metric_data("CompileTime")[:2] if met.metric_data("CompileTime") else None, flush=True)
print(o[0, ids.shape[1]:].cpu().tolist())
