#!/usr/bin/env python3
"""Compare load_causal_lm wall time: default vs TENSCUDA_LOADER_PREFETCH=1.

    python scripts/loader_bench_prefetch.py MODEL

Fresh process per mode. Writes stdout JSON line only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _py() -> str:
    v = ROOT / ".venv/bin/python"
    return str(v if v.is_file() else sys.executable)


def _one(model: str, prefetch: bool) -> float:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if prefetch:
        env["TENSCUDA_LOADER_PREFETCH"] = "1"
    code = f"""
import json, time, torch, tenscuda
from tenscuda.loader import load_causal_lm
t0 = time.time()
load_causal_lm({model!r}, dtype=torch.bfloat16, verbose=False)
print(json.dumps({{"prefetch": {prefetch!r}, "load_s": round(time.time()-t0, 2)}}))
"""
    p = subprocess.run([_py(), "-c", code], cwd=ROOT, env=env, text=True, capture_output=True)
    for line in reversed(p.stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)["load_s"]
    raise RuntimeError(p.stderr[-300:] or p.stdout[-300:])


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base = _one(model, False)
    pre = _one(model, True)
    out = {"model": model, "load_s_default": base, "load_s_prefetch": pre}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()