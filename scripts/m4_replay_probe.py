#!/usr/bin/env python3
"""M4 baseline: per-decode-step trace time and XLA compile count after warmup.

    python scripts/m4_replay_probe.py MODEL [N_STEPS=32]

Writes JSON to results/m4_replay_probe_<slug>.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def xla_compiles() -> int:
    import torch_xla.debug.metrics as met

    row = met.metric_data("CompileTime")
    return int(row[0]) if row else 0


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    model_id = sys.argv[1]
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tenscuda.generate import static_greedy_generate

    if not tenscuda.is_active:
        sys.exit("tenscuda: no TT device")

    tok = AutoTokenizer.from_pretrained(model_id)
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    if tok.chat_template:
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    else:
        ids = tok(msgs[0]["content"], return_tensors="pt").input_ids

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    stats: dict = {}
    resident = __import__("os").environ.get("TENSCUDA_RESIDENT_DECODE_IO", "0") == "1"

    t0 = time.time()
    static_greedy_generate(
        m, ids, max_new_tokens=n_steps, eos_token_id=tok.eos_token_id, stats=stats,
    )
    total_s = time.time() - t0

    steps = stats.get("step_seconds", [])
    prefill_s = round(steps[0], 2) if steps else None
    decode_steps = steps[1:] if len(steps) > 1 else []
    warm_ms = [round(1000 * s, 1) for s in decode_steps]
    per_step_compiles = stats.get("xla_compiles_per_step", [])

    payload = {
        "model": model_id,
        "n_steps": n_steps,
        "resident_decode_io": resident,
        "total_s": round(total_s, 2),
        "prefill_s": prefill_s,
        "decode_step_ms": warm_ms,
        "mean_decode_ms_steps_2_end": round(sum(warm_ms[1:]) / max(len(warm_ms) - 1, 1), 1) if len(warm_ms) > 1 else None,
        "xla_compiles_per_decode_step": per_step_compiles,
        "xla_compiles_delta_last_step": (
            per_step_compiles[-1] - per_step_compiles[0]
            if len(per_step_compiles) >= 2
            else None
        ),
    }
    out = ROOT / "results" / f"m4_replay_probe_{slug(model_id)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()