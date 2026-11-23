#!/usr/bin/env python3
"""A/B decode: compile unlock vs compile + TENSCUDA_NATIVE_OPS=sdpa,rms_norm.

Each variant runs in a fresh subprocess (env before import tenscuda). Writes
results/m4_native_compile_ab_<model-short>.json.

  PYTHONPATH=. .venv/bin/python scripts/m4_native_compile_ab.py MODEL [N]
  PYTHONPATH=. .venv/bin/python scripts/m4_native_compile_ab.py unsloth/Meta-Llama-3.1-8B-Instruct 32
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "compile": {
        "TENSCUDA_COMPILE": "1",
        "TENSCUDA_NATIVE_OPS": "",
    },
    "compile_native": {
        "TENSCUDA_COMPILE": "1",
        "TENSCUDA_NATIVE_OPS": "sdpa,rms_norm",
    },
}


def _child(label: str, model_id: str, n: int) -> None:
    cfg = VARIANTS[label]
    os.environ["TENSCUDA_COMPILE"] = cfg["TENSCUDA_COMPILE"]
    os.environ.pop("TENSCUDA_NATIVE_OPS", None)
    if cfg.get("TENSCUDA_NATIVE_OPS"):
        os.environ["TENSCUDA_NATIVE_OPS"] = cfg["TENSCUDA_NATIVE_OPS"]
    else:
        os.environ.pop("TENSCUDA_NATIVE_OPS", None)

    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tenscuda.generate import static_greedy_generate

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
    warm_ms = round(1000 * sum(warm) / max(len(warm), 1), 1) if warm else None
    warm_tok_s = round(len(warm) / sum(warm), 2) if warm and sum(warm) > 0 else None
    print(
        "RESULT:"
        + json.dumps(
            {
                "label": label,
                "tokens": out[0, ids.shape[1] :].cpu().tolist(),
                "warm_ms": warm_ms,
                "warm_tok_s": warm_tok_s,
                "native_ops": os.environ.get("TENSCUDA_NATIVE_OPS") or None,
            }
        ),
        flush=True,
    )


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    if len(sys.argv) > 3:
        label = sys.argv[3]
        if label not in VARIANTS:
            raise SystemExit(f"unknown variant {label}; use {list(VARIANTS)}")
        _child(label, model_id, n)
        return

    rows = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'scripts'}:{env.get('PYTHONPATH', '')}"
    timeout = int(os.environ.get("TENSCUDA_NATIVE_COMPILE_AB_TIMEOUT", "7200"))
    script = Path(__file__).resolve()
    for label in VARIANTS:
        print(f"=== {label} ===", flush=True)
        cmd = [sys.executable, str(script), model_id, str(n), label]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            rows[label] = {"error": (proc.stderr or proc.stdout)[-800:]}
            continue
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT:"):
                payload = json.loads(line[len("RESULT:") :])
        rows[label] = payload
        print(payload, flush=True)

    base = (rows.get("compile") or {}).get("tokens")
    nat = (rows.get("compile_native") or {}).get("tokens")
    out = {
        "model": model_id,
        "n": n,
        "variants": VARIANTS,
        "rows": rows,
    }
    if base and nat:
        pref = sum(1 for a, b in zip(base, nat) if a == b)
        out["compile_native_vs_compile"] = f"{pref}/{len(base)}"
        out["compile_native_full"] = nat == base
        c_ms = (rows.get("compile") or {}).get("warm_ms")
        n_ms = (rows.get("compile_native") or {}).get("warm_ms")
        if c_ms and n_ms:
            out["delta_pct_faster"] = round(100.0 * (c_ms - n_ms) / c_ms, 1)
    short = model_id.split("/")[-1]
    dest = ROOT / "results" / f"m4_native_compile_ab_{short}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()