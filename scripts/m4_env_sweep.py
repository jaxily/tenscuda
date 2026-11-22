#!/usr/bin/env python3
"""Sweep TensCUDA / compile env vars; parse trace_cost and replay probes (rock only).

    python scripts/m4_env_sweep.py

Writes results/m4_env_sweep_<timestamp>.json. Does not change defaults.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def _py() -> str:
    venv = ROOT / ".venv/bin/python"
    return str(venv if venv.is_file() else sys.executable)


def _trace_cost(extra: dict[str, str]) -> dict:
    e = {**os.environ, **extra, "PYTHONPATH": str(ROOT)}
    p = subprocess.run(
        [_py(), "scripts/trace_cost.py", MODEL, "default"],
        cwd=ROOT,
        env=e,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    m = re.search(r"TRACE (\{.*\})", p.stdout)
    if m:
        return json.loads(m.group(1))
    return {"error": p.stdout[-800:]}


def _replay(extra: dict[str, str]) -> dict:
    e = {**os.environ, **extra, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        [_py(), "scripts/m4_replay_probe.py", MODEL, "16"],
        cwd=ROOT,
        env=e,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    suffix = "_pause" if extra.get("TENSCUDA_DECODE_PAUSE_MODES") == "1" else ""
    path = ROOT / "results" / f"m4_replay_probe_{_slug(MODEL)}{suffix}.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"error": "missing replay json"}


def main() -> None:
    # Do not ``import tenscuda`` here: it opens the PJRT device and child
    # ``trace_cost`` / ``m4_replay_probe`` subprocesses block on CHIP_IN_USE.

    sweeps: list[tuple[str, dict[str, str]]] = [
        ("default", {}),
        ("hifi2", {"TENSCUDA_MATH_FIDELITY": "hifi2"}),
        ("lofi", {"TENSCUDA_MATH_FIDELITY": "lofi"}),
        ("fp32_acc_off", {"TENSCUDA_FP32_ACC": "0"}),
        ("exact_softmax_off", {"TENSCUDA_EXACT_SOFTMAX": "0"}),
        ("migration_log_off", {"TENSCUDA_MIGRATION_LOG": "0"}),
        ("decode_pause_modes", {"TENSCUDA_DECODE_PAUSE_MODES": "1"}),
        ("xla_dynamo_disable", {"XLA_DYNAMO_DISABLE": "1"}),
        ("thin_decode", {"TENSCUDA_THIN_DECODE": "1"}),
    ]
    rows = []
    for name, extra in sweeps:
        print(f"sweep {name}...", flush=True)
        t0 = time.time()
        rows.append({
            "name": name,
            "env": extra,
            "trace_cost": _trace_cost(extra),
            "replay": _replay(extra),
            "seconds": round(time.time() - t0, 1),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = ROOT / "results" / f"m4_env_sweep_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "sweeps": rows}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()