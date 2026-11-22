#!/usr/bin/env python3
"""E3: compare StaticCache pad_to buckets for compile count and first-run time.

    python scripts/m4_pad_bucket.py [MODEL]

Writes results/m4_pad_bucket_<slug>.json (fresh process per pad_to).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def _py() -> str:
    v = ROOT / ".venv/bin/python"
    return str(v if v.is_file() else sys.executable)


def _one_run(model: str, pad_to: int) -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.run(
        [_py(), "scripts/m4_pad_bucket_worker.py", model, str(pad_to)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"pad_to": pad_to, "error": p.stdout[-500:]}


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    rows = []
    for pt in (64, 128, 256):
        print(f"pad_to={pt}...", flush=True)
        rows.append(_one_run(model, pt))

    out = ROOT / "results" / f"m4_pad_bucket_{slug(model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "buckets": rows}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()