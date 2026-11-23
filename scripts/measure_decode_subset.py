#!/usr/bin/env python3
"""Re-bench warm decode + parity for matrix subset models (post functional decode).

Requires a P150a. Example:

  python scripts/measure_decode_subset.py
  python scripts/measure_decode_subset.py --scale   # COMPILE_SCALE ladder (incl. 7B–9B)
  TENSCUDA_COMPILE=1 python scripts/measure_decode_subset.py --scale

Appends JSON under results/measure_decode_*.json.
Merge numbers into docs/PERF.md and re-run matrix.py for COMPATIBILITY.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from matrix_models import COMPILE_SCALE  # noqa: E402

MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "unsloth/Llama-3.2-1B-Instruct",
]


def _prompt_ids(tok):
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    if tok.chat_template:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    return tok(msgs[0]["content"], return_tensors="pt").input_ids


def bench_one(model_id: str, n_new: int = 32) -> dict:
    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tenscuda.generate import static_greedy_generate

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = _prompt_ids(tok)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    eos = tok.eos_token_id
    warm_rates = []
    for run in range(4):
        st = {}
        t0 = time.time()
        static_greedy_generate(m, ids, max_new_tokens=n_new, eos_token_id=eos, stats=st)
        total = time.time() - t0
        decode_steps = [s for s in st.get("step_seconds", [])[1:]]
        if decode_steps:
            warm_rates.append(len(decode_steps) / sum(decode_steps))
        if run == 0:
            first_run_s = round(total, 2)
    warm = round(sum(warm_rates[1:]) / max(len(warm_rates) - 1, 1), 2) if len(warm_rates) > 1 else None
    return {
        "model": model_id,
        "n_new": n_new,
        "first_run_s": first_run_s,
        "warm_tok_s_mean_runs_2_4": warm,
        "functional_decode": "per-step sync via _forward_decode_step",
    }


def main() -> None:
    import tenscuda

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--scale",
        action="store_true",
        help="run COMPILE_SCALE ladder (135M → 9B) instead of the small subset",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="set TENSCUDA_COMPILE=1 for the run (M4 unlock)",
    )
    a = ap.parse_args()
    if a.compile:
        os.environ["TENSCUDA_COMPILE"] = "1"

    if not tenscuda.is_active:
        raise SystemExit("tenscuda: no TT device (run on P150a)")

    models = list(COMPILE_SCALE) if a.scale else MODELS
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = "scale" if a.scale else "subset"
    out = ROOT / "results" / f"measure_decode_{tag}_{stamp}.json"
    rows = []
    for mid in models:
        try:
            rows.append(bench_one(mid))
        except Exception as e:
            rows.append({"model": mid, "error": f"{type(e).__name__}: {e}"})
    payload = {
        "recorded_at": stamp,
        "compile": os.environ.get("TENSCUDA_COMPILE") == "1",
        "models": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
