import json
from pathlib import Path

from parse_tt_speed_logs import merge_showcase, parse_metal_b32, parse_vllm_offline, parse_vllm_serve


def test_parse_metal_b32():
    log = (
        "Average speed: 29.66ms @ 33.72 tok/s/user (1078.88 tok/s throughput)\n"
        "TTFT: 14.31 ms\nPASSED"
    )
    row = parse_metal_b32(log)
    assert row["tok_s_throughput"] == 1078.88
    assert row["vs_cpu_multiple"] == 353.7


def test_parse_vllm_offline():
    log = (
        "Avg generation throughput: 926.9 tokens/s\n"
        "Processed prompts: 100%|...| output: 908.56 toks/s"
    )
    row = parse_vllm_offline(log)
    assert row["engine_peak_generation_tok_s"] == 926.9
    assert row["tqdm_output_tok_s_final"] == 908.56


def test_parse_vllm_serve():
    log = (
        "Output token throughput (tok/s):         137.27\n"
        "Peak output token throughput (tok/s):    160.00\n"
    )
    row = parse_vllm_serve(log)
    assert row["output_token_throughput_tok_s"] == 137.27
    assert row["peak_output_token_throughput_tok_s"] == 160.0


def test_export_vllm_serve_json(tmp_path: Path, monkeypatch):
    import export_vllm_serve_json as exporter

    log = tmp_path / "bench.log"
    log.write_text(
        "Output token throughput (tok/s):         876.0\n"
        "Request throughput (req/s):              12.3\n"
    )
    out = tmp_path / "out.json"
    monkeypatch.setattr(exporter, "REPO", tmp_path)
    monkeypatch.setattr(
        exporter.sys,
        "argv",
        ["export_vllm_serve_json.py", str(log), "test-model"],
    )
    # Patch results dir via REPO
    (tmp_path / "results").mkdir()
    exporter.main()
    written = list((tmp_path / "results").glob("vllm_serve_b32_speed_*.json"))
    assert len(written) == 1
    doc = json.loads(written[0].read_text())
    assert doc["output_token_throughput_tok_s"] == 876.0


def test_merge_showcase(tmp_path: Path):
    doc = merge_showcase(
        model="m",
        host="h",
        runs=[{"path": "metal_b32", "tok_s_throughput": 1000.0}],
        logs={},
    )
    assert doc["headline_aggregate_tok_s"] == 1000.0