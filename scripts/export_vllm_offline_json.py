#!/usr/bin/env python3
"""Write results/vllm_offline_b32_speed_<ts>.json from a vLLM offline log."""
from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from parse_tt_speed_logs import CPU_REF_TOK_S, parse_vllm_offline  # noqa: E402


def main() -> None:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_offline_b32_speed.log")
    model = "unsloth/Meta-Llama-3.1-8B-Instruct"
    row = parse_vllm_offline(log.read_text(errors="replace"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    doc = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "path": "vllm-tt-plugin offline_inference_tt",
        "model": model,
        "max_seqs_in_batch": 32,
        "max_tokens": 128,
        "num_prompts": 256,
        "tqdm_output_tok_s_final": row.get("tqdm_output_tok_s_final"),
        "engine_peak_generation_tok_s": row.get("engine_peak_generation_tok_s"),
        "vs_cpu_tok_s": CPU_REF_TOK_S,
        "vs_cpu_multiple_tqdm": round(row["tqdm_output_tok_s_final"] / CPU_REF_TOK_S, 1)
        if row.get("tqdm_output_tok_s_final")
        else None,
        "vs_cpu_multiple_peak": round(row["engine_peak_generation_tok_s"] / CPU_REF_TOK_S, 1)
        if row.get("engine_peak_generation_tok_s")
        else None,
        "log": str(log),
    }
    out = REPO / "results" / f"vllm_offline_b32_speed_{ts}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()