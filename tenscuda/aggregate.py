"""Aggregate throughput via concurrent vLLM-TT HTTP (target ~900+ tok/s on P150).

Requires ``TENSCUDA_FAST=aggregate`` (or ``delegate``) and a warm vLLM-TT serve with
``max_num_seqs=32`` — see ``scripts/run_tt_serve_fast.sh``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tenscuda.fast_router import default_aggregate_concurrency, is_fast_eligible
from tenscuda.vllm_delegate import (
    AggregateResult,
    VllmDelegateError,
    delegate_aggregate,
    delegate_error_hint,
    probe_vllm,
)

DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
SCHEMA = "tenscuda-aggregate/1"


def run_aggregate(
    model_id: str,
    *,
    num_prompts: int | None = None,
    concurrency: int | None = None,
    max_new_tokens: int = 128,
    prompts: list[str] | None = None,
) -> AggregateResult:
    if not is_fast_eligible(model_id):
        raise ValueError(f"model {model_id!r} is not on the vLLM fast allowlist")
    if not probe_vllm():
        raise VllmDelegateError(delegate_error_hint(VllmDelegateError("vLLM not reachable")))
    return delegate_aggregate(
        model_id=model_id,
        prompts=prompts,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
        concurrency=concurrency,
    )


def _result_dict(res: AggregateResult) -> dict:
    return {
        "schema": SCHEMA,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ"),
        "model_id": res.model_id,
        "served_model": res.served_model,
        "num_prompts": res.num_prompts,
        "concurrency": res.concurrency,
        "max_new_tokens": res.max_new_tokens,
        "wall_seconds": round(res.wall_seconds, 4),
        "total_new_tokens": res.total_new_tokens,
        "aggregate_tok_s": round(res.aggregate_tok_s, 2),
        "errors": list(res.errors),
        "TENSCUDA_FAST": os.environ.get("TENSCUDA_FAST", ""),
        "vllm_base_url": os.environ.get("TENSCUDA_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Saturate local vLLM-TT with concurrent completions (aggregate tok/s).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id (allowlist)")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts (default: concurrency)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"Parallel HTTP workers (default: {default_aggregate_concurrency()})",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Write result JSON (default: results/tenscuda_aggregate_<utc>.json)",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation before writing JSON")
    args = parser.parse_args(argv)

    try:
        res = run_aggregate(
            args.model,
            num_prompts=args.num_prompts,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
        )
    except (VllmDelegateError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    payload = _result_dict(res)
    print(
        f"aggregate: {payload['aggregate_tok_s']:.1f} tok/s "
        f"({res.total_new_tokens} tokens in {res.wall_seconds:.2f}s, "
        f"{res.num_prompts} prompts, concurrency={res.concurrency})"
    )
    if res.errors:
        print(f"warn: {len(res.errors)} request(s) failed (partial throughput)")

    out = args.json_out.strip()
    if not out:
        root = Path(__file__).resolve().parents[1]
        out = str(root / "results" / f"tenscuda_aggregate_{payload['timestamp']}.json")
    if args.yes or input(f"Write {out}? [y/N] ").strip().lower() in {"y", "yes"}:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())