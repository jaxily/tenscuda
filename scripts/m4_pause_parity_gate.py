#!/usr/bin/env python3
"""kv 8/8 + trace_cost token match for default vs TENSCUDA_DECODE_PAUSE_MODES=1.

    python scripts/m4_pause_parity_gate.py MODEL

Does not import tenscuda in the parent (subprocess only).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def _py() -> str:
    v = ROOT / ".venv/bin/python"
    return str(v if v.is_file() else sys.executable)


def _kv(env: dict[str, str], model: str, n: int = 8) -> dict:
    e = {**os.environ, **env, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        [_py(), "scripts/kv_diagnose.py", model, str(n)],
        cwd=ROOT,
        env=e,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    path = ROOT / "results" / f"kv_diagnose_{_slug(model)}.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"error": f"missing {path}"}


def _trace_cost(env: dict[str, str], model: str) -> dict:
    e = {**os.environ, **env, "PYTHONPATH": str(ROOT)}
    p = subprocess.run(
        [_py(), "scripts/trace_cost.py", model, "default"],
        cwd=ROOT,
        env=e,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    m = re.search(r"TRACE (\{.*\})", p.stdout)
    return json.loads(m.group(1)) if m else {"error": p.stdout[-200:]}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    model = sys.argv[1]
    base_env: dict[str, str] = {}
    pause_env = {"TENSCUDA_DECODE_PAUSE_MODES": "1"}
    kv_pause = _kv(pause_env, model)
    tr_b = _trace_cost(base_env, model)
    tr_p = _trace_cost(pause_env, model)
    toks_b = tr_b.get("tokens", [])
    toks_p = tr_p.get("tokens", [])
    agree = 0
    for a, b in zip(toks_b, toks_p):
        if a != b:
            break
        agree += 1
    report = {
        "model": model,
        "kv_parity_resident_vs_safe": kv_pause.get("parity_resident_vs_safe"),
        "trace_tokens_agree_prefix": f"{agree}/{min(len(toks_b), len(toks_p))}",
        "tokens_identical": toks_b == toks_p,
        "trace_ms_baseline": tr_b.get("trace_ms"),
        "trace_ms_pause": tr_p.get("trace_ms"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()