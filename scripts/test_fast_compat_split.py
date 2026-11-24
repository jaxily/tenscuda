#!/usr/bin/env python3
"""Test fast vs compat split: decision matrix + optional on-card compat bench.

Offline (no card):
  python scripts/test_fast_compat_split.py --from-artifacts

On card (short compat measure for one model):
  PYTHONPATH=. .venv/bin/python scripts/test_fast_compat_split.py --bench SmolLM2-135M
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_fast_router():
    import importlib.util

    path = ROOT / "tenscuda" / "fast_router.py"
    spec = importlib.util.spec_from_file_location("tenscuda_fast_router", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_fr = _load_fast_router()
decide_route = _fr.decide_route
fast_mode = _fr.fast_mode
is_fast_eligible = _fr.is_fast_eligible


COMBOS = [
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 1, True, "advisory"),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 1, True, "enforce"),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 1, True, "delegate"),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 1, True, "off"),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 32, True, "advisory"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", 1, True, "advisory"),
    ("unsloth/gemma-2-9b-it", 1, True, "advisory"),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 1, False, "advisory"),
]


def _load_fast_tok_s() -> float | None:
    runs = sorted((ROOT / "results").glob("vllm_offline_b32_speed_*.json"))
    for p in reversed(runs):
        doc = json.loads(p.read_text())
        v = doc.get("tqdm_output_tok_s_final") or doc.get("engine_peak_generation_tok_s")
        if v:
            return float(v)
    return None


def _load_compat_tok_s(model_suffix: str) -> float | None:
    compat = ROOT / "docs" / "COMPATIBILITY.md"
    if not compat.is_file():
        return None
    for line in compat.read_text().splitlines():
        if model_suffix in line and "|" in line:
            parts = [c.strip() for c in line.strip("|").split("|")]
            if len(parts) >= 6 and "Llama-3.1-8B" in line:
                try:
                    return float(parts[5].split()[0])
                except ValueError:
                    pass
    p = ROOT / "results" / "matrix" / "Meta-Llama-3.1-8B-Instruct.json"
    if p.is_file():
        doc = json.loads(p.read_text())
        rows = doc.get("tt", {}).get("runs", [])
        for row in rows:
            if row.get("decode_tok_per_s"):
                return float(row["decode_tok_per_s"])
    return None


def bench_compat(model_id: str, tokens: int = 16) -> dict:
    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tenscuda.generate import static_greedy_generate

    os.environ.setdefault("TENSCUDA_FAST", "0")
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok("Hi", return_tensors="pt").input_ids
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    warm = []
    for _ in range(2):
        st = {}
        t0 = time.time()
        static_greedy_generate(m, ids, max_new_tokens=tokens, eos_token_id=tok.eos_token_id, stats=st)
        dec = st.get("step_seconds", [])[1:]
        if dec:
            warm.append(len(dec) / sum(dec))
    return {
        "compat_warm_tok_s": round(sum(warm) / len(warm), 2) if warm else None,
        "route_at_bench": decide_route(model_id=model_id, batch_size=1, greedy=True).route,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-artifacts", action="store_true", help="Skip card; use results/ + COMPATIBILITY")
    ap.add_argument("--bench", metavar="MODEL", help="Run short compat bench on card")
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()

    decisions = []
    for model_id, bsz, greedy, mode in COMBOS:
        d = decide_route(model_id=model_id, batch_size=bsz, greedy=greedy, fast_mode_value=mode)
        decisions.append(
            {
                "model_id": model_id,
                "batch_size": bsz,
                "greedy": greedy,
                "TENSCUDA_FAST": mode,
                "eligible": is_fast_eligible(model_id),
                "route": d.route,
                "reason": d.reason,
            }
        )

    fast_tok = _load_fast_tok_s()
    compat_tok = _load_compat_tok_s("Meta-Llama-3.1-8B")
    measured = {
        "fast_path_aggregate_tok_s_vllm_offline_b32": fast_tok,
        "compat_path_batch1_tok_s_llama8b_matrix": compat_tok,
        "ratio_fast_over_compat": round(fast_tok / compat_tok, 1) if fast_tok and compat_tok else None,
    }
    if args.bench:
        measured["bench"] = bench_compat(args.bench, tokens=args.tokens)

    doc = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "fast_mode_env": fast_mode(),
        "decision_matrix": decisions,
        "measured": measured,
        "notes": (
            "route=delegate when TENSCUDA_FAST=delegate|enforce|auto|vllm; "
            "generate() uses vLLM OpenAI API and skips TT-XLA decode when serve is up."
        ),
    }
    out = ROOT / "results" / f"fast_compat_split_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M')}Z.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(measured, indent=2))
    print(f"decision rows: {len(decisions)}")
    print(out)

    if args.from_artifacts or not args.bench:
        assert decisions[0]["route"] == "fast"
        assert decisions[1]["route"] == "delegate"
        assert decisions[5]["route"] == "compat"


if __name__ == "__main__":
    main()