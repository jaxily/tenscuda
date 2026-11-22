#!/usr/bin/env python3
"""Warm decode cProfile summary for M4 (where does trace time go?).

    python scripts/m4_trace_profile.py MODEL [WARM_STEPS=10]

Writes results/m4_trace_profile_<slug>.json
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def _prompt_ids(tok):
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    if tok.chat_template:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    return tok(msgs[0]["content"], return_tensors="pt").input_ids


def _warm_then_profile(model, ids, warm_steps: int, eos):
    from tenscuda.generate import static_greedy_generate

    static_greedy_generate(model, ids, max_new_tokens=warm_steps, eos_token_id=eos)
    prof = cProfile.Profile()
    prof.enable()
    t0 = time.time()
    static_greedy_generate(model, ids, max_new_tokens=warm_steps, eos_token_id=eos)
    wall_s = time.time() - t0
    prof.disable()
    return prof, wall_s


def _top_stats(prof, n: int = 25) -> list[dict]:
    buf = io.StringIO()
    ps = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
    ps.print_stats(n)
    lines = buf.getvalue().splitlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("ncalls") or line.startswith("Ordered by"):
            continue
        if "/" in line[:20] or line[0].isdigit():
            out.append({"line": line})
        if len(out) >= n:
            break
    return out


def _tag_hits(stats: list[dict]) -> dict[str, int]:
    tags = {"tt_torch": 0, "transformers": 0, "tenscuda": 0, "torch_xla": 0, "other": 0}
    for row in stats:
        s = row.get("line", "").lower()
        if "tt_torch" in s or "torch_overrides" in s:
            tags["tt_torch"] += 1
        elif "transformers" in s:
            tags["transformers"] += 1
        elif "tenscuda" in s:
            tags["tenscuda"] += 1
        elif "torch_xla" in s or "xla" in s:
            tags["torch_xla"] += 1
        else:
            tags["other"] += 1
    return tags


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    model_id = sys.argv[1]
    warm_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not tenscuda.is_active:
        sys.exit("tenscuda: no TT device")

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = _prompt_ids(tok)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    eos = tok.eos_token_id

    prof, wall_s = _warm_then_profile(m, ids, warm_steps, eos)
    top = _top_stats(prof, 30)
    payload = {
        "model": model_id,
        "warm_steps": warm_steps,
        "profile_wall_s": round(wall_s, 2),
        "ms_per_step_approx": round(1000 * wall_s / max(warm_steps, 1), 1),
        "top_cumulative_stats": top,
        "stack_tags_in_top30": _tag_hits(top),
    }
    out = ROOT / "results" / f"m4_trace_profile_{slug(model_id)}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "top_cumulative_stats"}, indent=2))
    print("top5:")
    for row in top[:5]:
        print(row["line"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()