#!/usr/bin/env python3
"""Parse tt-metal / vLLM-TT speed logs into a showcase JSON artifact."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CPU_REF_TOK_S = float(os.environ.get("SHOWCASE_CPU_REF_TOK_S", "3.05"))

METAL_PERF = re.compile(
    r"(?P<avg_ms>[\d.]+)\s*ms\s*@\s*(?P<tok_u>[\d.]+)\s*tok/s/user\s*\((?P<thr>[\d.]+)\s*tok/s(?:\s+throughput)?\)",
    re.I,
)
METAL_TTFT = re.compile(r"TTFT[:\s]+(?P<ttft>[\d.]+)\s*ms", re.I)
METAL_RESULT = re.compile(r"\b(?P<res>PASSED|FAILED)\b")

VLLM_OFF_TQDM = re.compile(r"(?P<rate>[\d.]+)\s*tok/s", re.I)
VLLM_OFF_TQDM_OUTPUT = re.compile(r"output:\s*(?P<rate>[\d.]+)\s*toks/s", re.I)
VLLM_OFF_ENGINE_AVG = re.compile(
    r"Avg generation throughput:\s*(?P<peak>[\d.]+)\s*tokens/s",
    re.I,
)
VLLM_OFF_PEAK = re.compile(
    r"peak\s+generation\s+tok/s[:\s]+(?P<peak>[\d.]+)",
    re.I | re.MULTILINE,
)
VLLM_OFF_PEAK_ALT = re.compile(
    r"generation\s+tok/s.*?['\"]?(?P<peak>[\d.]+)",
    re.I,
)

VLLM_SERVE_TPS = re.compile(
    r"(?:Output\s+token\s+throughput|throughput).*?(?P<tps>[\d.]+)\s*tok/s",
    re.I,
)
VLLM_SERVE_REQ = re.compile(
    r"Request\s+throughput.*?(?P<rps>[\d.]+)\s*req/s",
    re.I,
)


def _read(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")


def parse_metal_b32(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"path": "metal_b32"}
    m = None
    for line in log_text.splitlines():
        if "average speed" in line.lower():
            m = METAL_PERF.search(line)
            if m:
                break
    if m is None:
        matches = list(METAL_PERF.finditer(log_text))
        m = matches[-1] if matches else None
    if m:
        out["avg_ms_per_step"] = float(m.group("avg_ms"))
        out["tok_s_per_user"] = float(m.group("tok_u"))
        out["tok_s_throughput"] = float(m.group("thr"))
    t = METAL_TTFT.search(log_text)
    if t:
        out["ttft_ms"] = float(t.group("ttft"))
    r = METAL_RESULT.search(log_text)
    if r:
        out["result"] = r.group("res")
    if "tok_s_throughput" in out:
        out["vs_cpu_tok_s"] = CPU_REF_TOK_S
        out["vs_cpu_multiple"] = round(out["tok_s_throughput"] / CPU_REF_TOK_S, 1)
    return out


def parse_vllm_offline(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"path": "vllm_offline"}
    peaks: list[float] = []
    for pat in (VLLM_OFF_PEAK, VLLM_OFF_PEAK_ALT, VLLM_OFF_ENGINE_AVG):
        for m in pat.finditer(log_text):
            try:
                peaks.append(float(m.group("peak")))
            except (IndexError, ValueError):
                pass
    if peaks:
        out["engine_peak_generation_tok_s"] = max(peaks)

    rates = [float(m.group("rate")) for m in VLLM_OFF_TQDM_OUTPUT.finditer(log_text)]
    if not rates:
        rates = [float(m.group("rate")) for m in VLLM_OFF_TQDM.finditer(log_text)]
    if rates:
        out["tqdm_output_tok_s_final"] = rates[-1]
        out["tqdm_samples_tail"] = rates[-5:]

    primary = out.get("engine_peak_generation_tok_s") or out.get("tqdm_output_tok_s_final")
    if primary:
        out["vs_cpu_tok_s"] = CPU_REF_TOK_S
        out["vs_cpu_multiple"] = round(float(primary) / CPU_REF_TOK_S, 1)
    return out


def parse_vllm_serve(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"path": "vllm_serve"}
    m = VLLM_SERVE_TPS.search(log_text)
    if m:
        out["output_token_throughput_tok_s"] = float(m.group("tps"))
    r = VLLM_SERVE_REQ.search(log_text)
    if r:
        out["request_throughput_req_s"] = float(r.group("rps"))
    if "output_token_throughput_tok_s" in out:
        out["vs_cpu_tok_s"] = CPU_REF_TOK_S
        out["vs_cpu_multiple"] = round(
            out["output_token_throughput_tok_s"] / CPU_REF_TOK_S, 1
        )
    return out


def merge_showcase(
    *,
    model: str,
    host: str,
    runs: list[dict[str, Any]],
    logs: dict[str, str],
) -> dict[str, Any]:
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "model": model,
        "cpu_reference_tok_s": CPU_REF_TOK_S,
        "runs": runs,
        "logs": logs,
        "headline_aggregate_tok_s": _headline(runs),
    }


def _headline(runs: list[dict[str, Any]]) -> float | None:
    best = 0.0
    for row in runs:
        for key in (
            "tok_s_throughput",
            "engine_peak_generation_tok_s",
            "tqdm_output_tok_s_final",
            "output_token_throughput_tok_s",
        ):
            v = row.get(key)
            if isinstance(v, (int, float)) and v > best:
                best = float(v)
    return best if best > 0 else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("HF_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct"))
    p.add_argument("--host", default=os.environ.get("SHOWCASE_HOST") or socket.gethostname())
    p.add_argument("--out", type=Path, help="Write JSON here (default: results/showcase_<ts>.json)")
    p.add_argument("--metal-log", type=Path, default=Path("/tmp/tt_metal_b32_speed.log"))
    p.add_argument("--vllm-offline-log", type=Path, default=Path("/tmp/vllm_offline_b32_speed.log"))
    p.add_argument("--vllm-serve-log", type=Path, default=Path("/tmp/vllm_bench_8b.log"))
    p.add_argument(
        "--vllm-json",
        type=Path,
        help="Import vLLM offline metrics from a prior results/vllm_offline_*.json",
    )
    p.add_argument("--skip-metal", action="store_true")
    p.add_argument("--skip-offline", action="store_true")
    p.add_argument("--skip-serve", action="store_true", default=True)
    p.add_argument("--include-serve", action="store_true", help="Parse vllm bench serve log")
    args = p.parse_args()

    if args.include_serve:
        args.skip_serve = False

    runs: list[dict[str, Any]] = []
    logs: dict[str, str] = {}

    def _nonempty(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    if not args.skip_metal and _nonempty(args.metal_log):
        logs["metal_b32"] = str(args.metal_log)
        runs.append(parse_metal_b32(_read(args.metal_log)))
    if not args.skip_offline and _nonempty(args.vllm_offline_log):
        logs["vllm_offline"] = str(args.vllm_offline_log)
        offline_row = parse_vllm_offline(_read(args.vllm_offline_log))
        if offline_row.get("tqdm_output_tok_s_final") or offline_row.get(
            "engine_peak_generation_tok_s"
        ):
            runs.append(offline_row)
    if (
        not any(r.get("path") == "vllm_offline" for r in runs)
        and args.vllm_json
        and args.vllm_json.is_file()
    ):
        logs["vllm_offline"] = str(args.vllm_json)
        prior = json.loads(args.vllm_json.read_text())
        row: dict[str, Any] = {"path": "vllm_offline", "source": "json"}
        if prior.get("tqdm_output_tok_s_final") is not None:
            row["tqdm_output_tok_s_final"] = prior["tqdm_output_tok_s_final"]
        if prior.get("engine_peak_generation_tok_s") is not None:
            row["engine_peak_generation_tok_s"] = prior["engine_peak_generation_tok_s"]
        primary = row.get("engine_peak_generation_tok_s") or row.get("tqdm_output_tok_s_final")
        if primary:
            row["vs_cpu_tok_s"] = CPU_REF_TOK_S
            row["vs_cpu_multiple"] = round(float(primary) / CPU_REF_TOK_S, 1)
        runs.append(row)
    if not args.skip_serve and _nonempty(args.vllm_serve_log):
        logs["vllm_serve"] = str(args.vllm_serve_log)
        runs.append(parse_vllm_serve(_read(args.vllm_serve_log)))

    if not runs:
        raise SystemExit("No logs parsed; pass existing log paths or run run_tt_speed_showcase.sh first")

    doc = merge_showcase(model=args.model, host=args.host, runs=runs, logs=logs)
    repo = Path(__file__).resolve().parents[1]
    out = args.out or repo / "results" / f"showcase_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M')}Z.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()