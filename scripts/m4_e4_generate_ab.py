#!/usr/bin/env python3
"""Decode A/B: eager vs TENSCUDA_NATIVE_OPS=rms_norm (E4 composite on XLA)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _child(label: str, model_id: str, n: int) -> None:
    import tenscuda  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tenscuda.generate import static_greedy_generate
    import torch

    os.environ.pop("TENSCUDA_NATIVE_OPS", None)
    if label == "native":
        os.environ["TENSCUDA_NATIVE_OPS"] = "rms_norm"
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
                "label": label,
                "tokens": out[0, ids.shape[1] :].cpu().tolist(),
                "warm_ms": round(1000 * sum(warm) / max(len(warm), 1), 1) if warm else None,
            }
        ),
        flush=True,
    )


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if len(sys.argv) > 3:
        _child(sys.argv[3], model_id, n)
        return

    rows = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT}/scripts:{env.get('PYTHONPATH', '')}"
    timeout = int(os.environ.get("TENSCUDA_E4_AB_TIMEOUT", "1800"))
    for label in ("eager", "native"):
        print(f"=== {label} ===", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), model_id, str(n), label]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            rows[label] = {"error": (proc.stderr or proc.stdout)[-400:]}
            continue
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT:"):
                payload = json.loads(line[len("RESULT:") :])
        rows[label] = payload
        print(payload, flush=True)

    base = (rows.get("eager") or {}).get("tokens")
    out = {"model": model_id, "n": n, "native_ops": "rms_norm", "rows": rows}
    if base and (rows.get("native") or {}).get("tokens"):
        nat = rows["native"]["tokens"]
        pref = sum(1 for a, b in zip(base, nat) if a == b)
        out["native_vs_eager"] = f"{pref}/{len(base)}"
        out["native_full"] = nat == base
    dest = ROOT / "results" / f"m4_e4_generate_ab_{model_id.split('/')[-1]}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()