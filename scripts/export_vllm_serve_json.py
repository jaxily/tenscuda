#!/usr/bin/env python3
"""Write results/vllm_serve_b32_speed_<ts>.json from a vllm bench serve log."""
from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from parse_tt_speed_logs import CPU_REF_TOK_S, parse_vllm_serve  # noqa: E402


def main() -> None:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_bench_8b.log")
    model = sys.argv[2] if len(sys.argv) > 2 else "unsloth/Meta-Llama-3.1-8B-Instruct"
    row = parse_vllm_serve(log.read_text(errors="replace"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    out_tps = row.get("output_token_throughput_tok_s")
    doc = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "path": "vllm-tt-plugin bench serve",
        "model": model,
        "max_num_seqs": 32,
        "num_prompts": 32,
        "output_token_throughput_tok_s": out_tps,
        "request_throughput_req_s": row.get("request_throughput_req_s"),
        "vs_cpu_tok_s": CPU_REF_TOK_S,
        "vs_cpu_multiple": round(float(out_tps) / CPU_REF_TOK_S, 1) if out_tps else None,
        "log": str(log),
    }
    out = REPO / "results" / f"vllm_serve_b32_speed_{ts}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()