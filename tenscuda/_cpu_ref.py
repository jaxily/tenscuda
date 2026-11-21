"""CPU greedy reference, run as a separate *script* (not via the tenscuda package, so the
TT device is never opened and host RAM is freed when it exits).

    python _cpu_ref.py MODEL DTYPE NEW_TOKENS OUT.json PROMPT

Plain greedy loop (no generation_config, no repetition penalty, no EOS stop), the same
as tenscuda.generate.static_greedy_generate. Saves per-step top-1/top-2 logit margins.
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def prompt_ids(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    if tok.chat_template:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
    return tok(prompt, return_tensors="pt").input_ids


def main():
    mid, dtype_s, n, out, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=getattr(torch, dtype_s), low_cpu_mem_usage=True).eval()
    ids = prompt_ids(tok, prompt)
    toks, margins, steps = [], [], []
    with torch.no_grad():
        t = time.time()
        o = m(input_ids=ids, use_cache=True)
        for i in range(n):
            top = o.logits[0, -1].float().topk(2)
            nxt = int(top.indices[0])
            toks.append(nxt)
            margins.append(round(float(top.values[0] - top.values[1]), 4))
            steps.append(time.time() - t)
            t = time.time()
            if i + 1 < n:
                o = m(input_ids=torch.tensor([[nxt]]), past_key_values=o.past_key_values, use_cache=True)
    dec = steps[1:]
    json.dump({"model": mid, "dtype": dtype_s, "new_tokens": n, "prompt_tokens": ids.shape[1],
               "decode_tok_per_s": round(len(dec) / sum(dec), 2) if dec else None,
               "tokens": toks, "margins": margins}, open(out, "w"))


if __name__ == "__main__":
    main()
