#!/usr/bin/env python3
"""M4: pinpoint where compile decode diverges (E5/modes: match 2/16).

Runs baseline vs TENSCUDA_COMPILE+CPU_INIT+PAUSE in one report:
per-step token, top1/top2 logits, and whether they match.

    PYTHONPATH=~/tenscuda .venv/bin/python scripts/m4_compile_diverge.py \\
        HuggingFaceTB/SmolLM2-135M-Instruct 8

Writes results/m4_compile_diverge_<slug>.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
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


def _step_loop(model, ids_cpu, n_new, *, compile: bool):
    from tenscuda.compile_unlock import compile_decode_steps

    return compile_decode_steps(model, ids_cpu, n_new, use_compile=compile)


def _child(label, model_id, n_new):
    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = _prompt_ids(tok)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    steps = _step_loop(m, ids, n_new, compile=(label == "compile"))
    print("RESULT:" + json.dumps({"label": label, "steps": steps}), flush=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    model_id = sys.argv[1]
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if len(sys.argv) > 3:
        _child(sys.argv[3], model_id, n_new)
        return

    rows = {}
    for label in ("baseline", "compile"):
        print(f"=== {label} ===", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), model_id, str(n_new), label]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], flush=True)
            rows[label] = {"error": proc.stderr[-500:]}
            continue
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT:"):
                payload = json.loads(line[len("RESULT:"):])
        rows[label] = payload

    # Align steps
    cmp = []
    b = rows.get("baseline", {}).get("steps") or []
    c = rows.get("compile", {}).get("steps") or []
    for i, (bs, cs) in enumerate(zip(b, c)):
        cmp.append({
            "i": i,
            "phase": bs["phase"],
            "base_tok": bs["token"],
            "comp_tok": cs["token"],
            "match": bs["token"] == cs["token"],
            "base_margin": bs["margin"],
            "comp_margin": cs["margin"],
            "base_top1_logit": bs["top1_logit"],
            "comp_top1_logit": cs["top1_logit"],
            "comp_ms": cs["ms"],
            "base_ms": bs["ms"],
        })
    first_bad = next((r["i"] for r in cmp if not r["match"]), None)
    out = {
        "model": model_id,
        "n_new": n_new,
        "first_diverge_step": first_bad,
        "compare": cmp,
        "baseline": rows.get("baseline"),
        "compile": rows.get("compile"),
    }
    dest = ROOT / "results" / f"m4_compile_diverge_{slug(model_id)}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"first_diverge_step={first_bad}")
    for r in cmp:
        mark = "OK" if r["match"] else "DIFF"
        print(f"  [{mark}] i={r['i']} base={r['base_tok']} comp={r['comp_tok']} "
              f"margins={r['base_margin']:.3f}/{r['comp_margin']:.3f} comp_ms={r['comp_ms']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
