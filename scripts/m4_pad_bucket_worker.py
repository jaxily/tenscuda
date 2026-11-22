"""One pad_to bucket (fresh process). Invoked by m4_pad_bucket.py."""
import json
import os
import sys
import time

import tenscuda  # noqa: F401
import torch
import torch_xla.debug.metrics as met
from transformers import AutoModelForCausalLM, AutoTokenizer

from tenscuda.generate import static_greedy_generate


def compiles() -> int:
    row = met.metric_data("CompileTime")
    return int(row[0]) if row else 0


def main() -> None:
    mid, pad_to = sys.argv[1], int(sys.argv[2])
    os.environ["TENSCUDA_PAD_TO"] = str(pad_to)
    tok = AutoTokenizer.from_pretrained(mid)
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"]
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    stats: dict = {}
    t0 = time.time()
    static_greedy_generate(m, ids, max_new_tokens=16, stats=stats)
    wall = time.time() - t0
    steps = stats.get("step_seconds") or []
    warm_ms = round(1000 * sum(steps[2:]) / len(steps[2:]), 1) if len(steps) > 2 else None
    print(
        json.dumps({
            "pad_to": pad_to,
            "max_cache_len": -(-(ids.shape[1] + 16) // pad_to) * pad_to,
            "first_run_s": round(wall, 2),
            "warm_step_ms": warm_ms,
            "xla_compiles": compiles(),
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()