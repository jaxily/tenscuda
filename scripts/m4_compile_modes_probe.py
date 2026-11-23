#!/usr/bin/env python3
"""M4 follow-up: does pausing torch-function modes make torch.compile parity-safe?

Hypothesis: TensCUDA / tt_torch __torch_function__ overrides poison Dynamo capture
(PERF §5 wrong from token 1). Capture + run with modes paused like B3a decode.

Also measures CPU-init StaticCache alone (no compile) for parity/speed.

    PYTHONPATH=~/tenscuda .venv/bin/python scripts/m4_compile_modes_probe.py \\
        HuggingFaceTB/SmolLM2-135M-Instruct 16

Each variant is a fresh process (device lock). Writes
results/m4_compile_modes_<slug>.json
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


def _run_child(label: str, model_id: str, n_new: int) -> dict:
    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tenscuda.generate import static_greedy_generate

    if not tenscuda.is_active:
        raise SystemExit("tenscuda: no TT device")

    # Map label -> env for static_greedy_generate
    env_flags = {
        "baseline": {},
        "cpu_init_only": {"TENSCUDA_STATIC_CACHE_CPU_INIT": "1"},
        "pause_only": {"TENSCUDA_DECODE_PAUSE_MODES": "1"},
        "cpu_init_pause": {
            "TENSCUDA_STATIC_CACHE_CPU_INIT": "1",
            "TENSCUDA_DECODE_PAUSE_MODES": "1",
        },
        "compile_pause": {
            "TENSCUDA_COMPILE": "1",
            "TENSCUDA_DECODE_PAUSE_MODES": "1",
            "TENSCUDA_STATIC_CACHE_CPU_INIT": "1",
        },
        "compile_pause_legacy": {
            "TENSCUDA_COMPILE": "1",
            "TENSCUDA_COMPILE_LEGACY": "1",
            "TENSCUDA_DECODE_PAUSE_MODES": "1",
            "TENSCUDA_STATIC_CACHE_CPU_INIT": "1",
        },
        "compile_pause_global": {
            # Pause modes for the WHOLE generate (prefill+decode) via monkeypatch
            "TENSCUDA_COMPILE": "1",
            "TENSCUDA_STATIC_CACHE_CPU_INIT": "1",
            "TENSCUDA_PAUSE_ALL_MODES": "1",
        },
    }
    flags = env_flags[label]
    for k, v in flags.items():
        if k == "TENSCUDA_PAUSE_ALL_MODES":
            continue
        os.environ[k] = v

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = _prompt_ids(tok)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()

    stats: dict = {}
    t0 = time.time()
    exited_modes: list = []
    if flags.get("TENSCUDA_PAUSE_ALL_MODES") == "1":
        # Exit torch-function modes for the entire generate (incl. Dynamo capture).
        import tenscuda._patch as patch

        mode = patch._S.originals.get("mode")
        if mode is not None:
            mode.__exit__(None, None, None)
            exited_modes.append(mode)
        try:
            import tt_torch.torch_overrides as overrides

            tt = overrides.torch_function_override
            if getattr(tt, "_entered", True):
                tt.__exit__(None, None, None)
                exited_modes.append(tt)
        except Exception:
            pass
        os.environ["TENSCUDA_DECODE_PAUSE_MODES"] = "0"  # already exited; avoid double-exit
    try:
        out = static_greedy_generate(
            m, ids, max_new_tokens=n_new, eos_token_id=None, stats=stats,
            compile=os.environ.get("TENSCUDA_COMPILE") == "1",
        )
    finally:
        for ctx in reversed(exited_modes):
            try:
                ctx.__enter__()
            except Exception:
                pass
    steps = stats.get("step_seconds", [])

    dec = steps[1:] if len(steps) > 1 else []
    # Prefer warm mean excluding first decode step if first is compile-heavy.
    warm = dec[1:] if len(dec) > 1 else dec
    tokens = [int(t) for t in out[0, ids.shape[1]:].detach().cpu().tolist()]
    return {
        "label": label,
        "tokens": tokens,
        "mean_decode_ms_all": round(1000 * sum(dec) / max(len(dec), 1), 1) if dec else None,
        "mean_decode_ms_warm": round(1000 * sum(warm) / max(len(warm), 1), 1) if warm else None,
        "first_decode_ms": round(1000 * dec[0], 1) if dec else None,
        "wall_s": round(time.time() - t0, 2),
        "flags": flags,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    model_id = sys.argv[1]
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    labels = [
        "baseline",
        "cpu_init_only",
        "pause_only",
        "cpu_init_pause",
        "compile_pause",
        "compile_pause_legacy",
        "compile_pause_global",
    ]
    if len(sys.argv) > 3:
        # child
        label = sys.argv[3]
        row = _run_child(label, model_id, n_new)
        print("RESULT:" + json.dumps(row), flush=True)
        return

    rows = []
    baseline = None
    for label in labels:
        print(f"=== {label} ===", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), model_id, str(n_new), label]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=720)
        except subprocess.TimeoutExpired:
            rows.append({"label": label, "error": "timeout 720s"})
            print("TIMEOUT", flush=True)
            continue
        if proc.stdout:
            print(proc.stdout[-1500:], flush=True)
        if proc.returncode != 0:
            rows.append({"label": label, "error": (proc.stderr or "")[-500:] or f"exit {proc.returncode}"})
            continue
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT:"):
                payload = json.loads(line[len("RESULT:"):])
        if not payload:
            rows.append({"label": label, "error": "no RESULT"})
            continue
        if label == "baseline":
            baseline = payload["tokens"]
        if baseline is not None and label != "baseline":
            match = 0
            for a, b in zip(baseline, payload["tokens"]):
                if a != b:
                    break
                match += 1
            payload["parity_vs_baseline"] = f"{match}/{len(baseline)}"
            payload["tokens_match"] = payload["tokens"] == baseline
        rows.append(payload)
        print(
            f"  match={payload.get('tokens_match')} parity={payload.get('parity_vs_baseline')} "
            f"warm_ms={payload.get('mean_decode_ms_warm')} wall={payload.get('wall_s')}",
            flush=True,
        )

    out = {"model": model_id, "n_new": n_new, "baseline_tokens": baseline, "variants": rows}
    dest = ROOT / "results" / f"m4_compile_modes_{slug(model_id)}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}")
    for r in rows:
        print({k: r.get(k) for k in ("label", "parity_vs_baseline", "tokens_match", "mean_decode_ms_warm", "error")})


if __name__ == "__main__":
    main()
