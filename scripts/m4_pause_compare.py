#!/usr/bin/env python3
"""A/B m4_replay_probe: default vs TENSCUDA_DECODE_PAUSE_MODES=1 (fresh subprocess each).

    python scripts/m4_pause_compare.py [MODEL ...]

Default models: SmolLM2-135M, TinyLlama-1.1B, Llama-3.2-1B.
Writes results/m4_pause_compare_<timestamp>.json
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

DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "unsloth/Llama-3.2-1B-Instruct",
]


def _py() -> str:
    v = ROOT / ".venv/bin/python"
    return str(v if v.is_file() else sys.executable)


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def _replay_subproc(model: str, pause: bool, n_steps: int = 32) -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if pause:
        env["TENSCUDA_DECODE_PAUSE_MODES"] = "1"
    else:
        env.pop("TENSCUDA_DECODE_PAUSE_MODES", None)
    subprocess.run(
        [_py(), "scripts/m4_replay_probe.py", model, str(n_steps)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    suffix = "_pause" if pause else ""
    path = ROOT / "results" / f"m4_replay_probe_{slug(model)}{suffix}.json"
    if not path.is_file():
        return {"error": f"missing {path}"}
    return json.loads(path.read_text())


def _delta_pct(baseline_ms: float | None, pause_ms: float | None) -> float | None:
    if baseline_ms is None or pause_ms is None or baseline_ms <= 0:
        return None
    return round(100.0 * (baseline_ms - pause_ms) / baseline_ms, 1)


def compare_one(model: str, n_steps: int = 32) -> dict:
    print(f"compare {model} baseline...", flush=True)
    t0 = time.time()
    baseline = _replay_subproc(model, pause=False, n_steps=n_steps)
    print(f"compare {model} pause...", flush=True)
    pause = _replay_subproc(model, pause=True, n_steps=n_steps)
    b_ms = baseline.get("mean_decode_ms_steps_2_end")
    p_ms = pause.get("mean_decode_ms_steps_2_end")
    return {
        "model": model,
        "n_steps": n_steps,
        "baseline_mean_decode_ms": b_ms,
        "pause_mean_decode_ms": p_ms,
        "delta_pct_faster": _delta_pct(b_ms, p_ms),
        "baseline_xla_delta": baseline.get("xla_compiles_delta_last_step"),
        "pause_xla_delta": pause.get("xla_compiles_delta_last_step"),
        "seconds": round(time.time() - t0, 1),
        "baseline": baseline,
        "pause": pause,
    }


def main() -> None:
    models = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS
    rows = [compare_one(m) for m in models]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = ROOT / "results" / f"m4_pause_compare_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": stamp, "comparisons": rows}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = [
        {
            "model": r["model"].split("/")[-1],
            "baseline_ms": r["baseline_mean_decode_ms"],
            "pause_ms": r["pause_mean_decode_ms"],
            "delta_pct": r["delta_pct_faster"],
        }
        for r in rows
    ]
    print(json.dumps({"summary": summary}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()