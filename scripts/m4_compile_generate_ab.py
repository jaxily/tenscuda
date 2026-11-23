#!/usr/bin/env python3
"""A/B: m4_compile_diverge loop vs static_greedy_generate(TENSCUDA_COMPILE=1)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if len(sys.argv) > 3:
        label = sys.argv[3]
        # child
        if label == "diverge":
            # reuse diverge child
            from m4_compile_diverge import _child
            _child("compile", model_id, n)
            return
        if label == "generate":
            import tenscuda  # noqa: F401
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from tenscuda.generate import static_greedy_generate

            os.environ["TENSCUDA_COMPILE"] = "1"
            tok = AutoTokenizer.from_pretrained(model_id)
            msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )["input_ids"]
            m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
            stats = {}
            out = static_greedy_generate(m, ids, max_new_tokens=n, eos_token_id=None, stats=stats)
            dec = stats.get("step_seconds", [])[1:]
            warm = dec[1:] if len(dec) > 1 else dec
            print(
                "RESULT:"
                + json.dumps(
                    {
                        "label": "generate",
                        "tokens": out[0, ids.shape[1] :].cpu().tolist(),
                        "warm_ms": round(1000 * sum(warm) / max(len(warm), 1), 1) if warm else None,
                    }
                ),
                flush=True,
            )
            return
        if label == "baseline":
            import tenscuda  # noqa: F401
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from tenscuda.generate import static_greedy_generate

            os.environ["TENSCUDA_COMPILE"] = "0"
            tok = AutoTokenizer.from_pretrained(model_id)
            msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )["input_ids"]
            m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
            stats = {}
            out = static_greedy_generate(m, ids, max_new_tokens=n, eos_token_id=None, stats=stats)
            dec = stats.get("step_seconds", [])[1:]
            warm = dec[1:] if len(dec) > 1 else dec
            print(
                "RESULT:"
                + json.dumps(
                    {
                        "label": "baseline",
                        "tokens": out[0, ids.shape[1] :].cpu().tolist(),
                        "warm_ms": round(1000 * sum(warm) / max(len(warm), 1), 1) if warm else None,
                    }
                ),
                flush=True,
            )
            return

    rows = {}
    for label in ("baseline", "diverge", "generate"):
        print(f"=== {label} ===", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), model_id, str(n), label]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}:{ROOT}/scripts:{env.get('PYTHONPATH', '')}"
        timeout_s = int(os.environ.get("TENSCUDA_COMPILE_AB_TIMEOUT", "3600"))
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            print(proc.stderr[-1500:], flush=True)
            rows[label] = {"error": proc.stderr[-400:]}
            continue
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT:"):
                payload = json.loads(line[len("RESULT:") :])
        # diverge child prints RESULT with steps — normalize to tokens
        if payload and "steps" in payload:
            payload = {
                "label": "diverge",
                "tokens": [s["token"] for s in payload["steps"]],
                "warm_ms": round(
                    sum(s["ms"] for s in payload["steps"] if s["phase"] == "decode" and s["ms"] < 500)
                    / max(1, sum(1 for s in payload["steps"] if s["phase"] == "decode" and s["ms"] < 500)),
                    1,
                ),
            }
        rows[label] = payload
        print(payload, flush=True)

    base = (rows.get("baseline") or {}).get("tokens")
    out = {"model": model_id, "n": n, "rows": rows}
    if base:
        for k in ("diverge", "generate"):
            toks = (rows.get(k) or {}).get("tokens")
            if not toks:
                continue
            pref = 0
            for a, b in zip(base, toks):
                if a != b:
                    break
                pref += 1
            out[f"{k}_vs_baseline"] = f"{pref}/{len(base)}"
            out[f"{k}_full"] = toks == base
    dest = ROOT / "results" / "m4_compile_generate_ab.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
