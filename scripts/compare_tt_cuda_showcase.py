#!/usr/bin/env python3
"""Merge Tenstorrent showcase JSON + optional NVIDIA ref into a deck-ready summary."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _pick_tt_aggregate(doc: dict[str, Any]) -> float | None:
    best = doc.get("headline_aggregate_tok_s")
    if isinstance(best, (int, float)):
        return float(best)
    for row in doc.get("runs", []):
        for key in ("tok_s_throughput", "engine_peak_generation_tok_s", "tqdm_output_tok_s_final"):
            v = row.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _pick_cuda_aggregate(doc: dict[str, Any]) -> float | None:
    for row in doc.get("runs", []):
        if row.get("path") != "vllm_serve":
            continue
        v = row.get("output_token_throughput_tok_s")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("showcase", type=Path, help="results/showcase_*.json from parse_tt_speed_logs")
    p.add_argument("--nvidia", type=Path, help="results/nvidia_ref_*.json (optional)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/showcase_tt_vs_cuda.json"),
    )
    args = p.parse_args()

    tt = json.loads(args.showcase.read_text())
    tt_agg = _pick_tt_aggregate(tt)
    cpu = float(tt.get("cpu_reference_tok_s", 3.05))

    summary: dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": tt.get("model"),
        "cpu_reference_tok_s": cpu,
        "tenstorrent": {
            "source": str(args.showcase),
            "host": tt.get("host"),
            "headline_aggregate_tok_s": tt_agg,
            "vs_cpu_multiple": round(tt_agg / cpu, 1) if tt_agg else None,
            "runs": tt.get("runs"),
        },
        "nvidia_cuda": None,
        "markdown_snippet": "",
    }

    cuda_agg: float | None = None
    if args.nvidia and args.nvidia.is_file():
        nv = json.loads(args.nvidia.read_text())
        cuda_agg = _pick_cuda_aggregate(nv)
        summary["nvidia_cuda"] = {
            "source": str(args.nvidia),
            "host": nv.get("host"),
            "output_token_throughput_tok_s": cuda_agg,
            "vs_cpu_multiple": round(cuda_agg / cpu, 1) if cuda_agg else None,
        }

    lines = [
        f"## {summary['model']} — serve-shaped bench (128 in / 128 out, 32 prompts)",
        "",
        "| Stack | Aggregate tok/s | vs CPU (~{:.1f}) |".format(cpu),
        "|---|---:|---:|",
    ]
    if tt_agg is not None:
        lines.append(f"| Tenstorrent P150 (showcase) | **{tt_agg:.0f}** | **~{tt_agg/cpu:.0f}×** |")
    if cuda_agg is not None:
        lines.append(f"| NVIDIA vLLM (same harness) | **{cuda_agg:.0f}** | **~{cuda_agg/cpu:.0f}×** |")
    elif summary["nvidia_cuda"] is None:
        lines.append("| NVIDIA vLLM | TBD — run `scripts/nvidia_vllm_bench_ref.sh` | — |")
    summary["markdown_snippet"] = "\n".join(lines) + "\n"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.out)
    print(summary["markdown_snippet"])


if __name__ == "__main__":
    main()