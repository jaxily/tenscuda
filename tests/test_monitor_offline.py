"""Offline tests for the card monitor (no GTK, no Tenstorrent hardware required)."""
from __future__ import annotations

import json
import os

from monitor import activity, sysfs


def test_activity_emit_and_latest(tmp_path, monkeypatch):
    path = tmp_path / "activity.jsonl"
    monkeypatch.setattr(activity, "ACTIVITY_PATH", str(path))
    monkeypatch.setattr(activity, "ACTIVITY_DIR", str(tmp_path))
    activity.emit("loading", model="HuggingFaceTB/SmolLM2-135M-Instruct", params="135M")
    activity.emit("decoding", model="HuggingFaceTB/SmolLM2-135M-Instruct", tok_s=4.2)
    row = activity.latest(max_age_s=60)
    assert row is not None
    assert row["phase"] == "decoding"
    assert row["tok_s"] == 4.2
    text = activity.format_workload(row)
    assert "Decoding" in text
    assert "4.2 tok/s" in text
    assert "SmolLM2-135M-Instruct" in text
    assert activity.format_workload(None) == "Idle"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["phase"] == "loading"


def test_sysfs_snapshot_keys():
    snap = sysfs.snapshot()
    for key in ("temp_c", "power_w", "aiclk_mhz", "ram_used_gib", "ram_total_gib",
                "device_present", "card_type", "architecture", "kmd"):
        assert key in snap


def test_architecture_label():
    assert "Tensix" in sysfs.architecture_label("Blackhole P150a")
    assert sysfs.architecture_label(None) == "—"


def test_activity_emit_never_raises(monkeypatch):
    monkeypatch.setattr(activity, "ACTIVITY_DIR", "/proc/does-not-exist-tenscuda")
    monkeypatch.setattr(activity, "ACTIVITY_PATH", "/proc/does-not-exist-tenscuda/x.jsonl")
    activity.emit("idle")  # must not raise


def test_latest_tok_s(tmp_path, monkeypatch):
    path = tmp_path / "activity.jsonl"
    monkeypatch.setattr(activity, "ACTIVITY_PATH", str(path))
    monkeypatch.setattr(activity, "ACTIVITY_DIR", str(tmp_path))
    assert activity.latest_tok_s() is None
    activity.emit("decoding", model="HuggingFaceTB/SmolLM2-135M-Instruct", tok_s=17.2, params="135M")
    assert activity.latest_tok_s() == 17.2
    info = activity.latest_decode()
    assert info["model"] == "SmolLM2-135M-Instruct"
    assert info["params"] == "135M"
    activity.emit("idle", tok_s=17.2)
    assert activity.latest_tok_s() is None


def test_short_model_name_strips_hf_snapshot_hash():
    path = (
        "/home/jake/.cache/huggingface/hub/"
        "models--HuggingFaceTB--SmolLM2-135M-Instruct/"
        "snapshots/fe8a4eaabc123def4567890abcdef1234567890"
    )
    assert activity.short_model_name(path) == "SmolLM2-135M-Instruct"
    assert activity.short_model_name("fe8a4eaabc123def4567890abcdef1234567890") is None
    assert activity.short_model_name("HuggingFaceTB/SmolLM2-135M-Instruct") == "SmolLM2-135M-Instruct"


def test_format_workload_video_progress():
    text = activity.format_workload(
        {
            "phase": "denoising",
            "kind": "video",
            "model": "MiniMax-H3",
            "progress": 0.08,
            "step": 0,
            "total_steps": 8,
            "mode": "fl2va",
        }
    )
    assert "Denoising" in text
    assert "MiniMax-H3" in text
    assert "8%" in text
    assert "step 0/8" in text
    assert "fl2va" in text


def test_latest_chart_prefers_minimax(monkeypatch):
    monkeypatch.setattr(
        activity,
        "latest_workload",
        lambda max_age_s=120.0: {
            "phase": "denoising",
            "kind": "video",
            "model": "MiniMax-H3",
            "progress": 0.25,
            "step": 2,
            "total_steps": 8,
        },
    )
    chart = activity.latest_chart()
    assert chart is not None
    assert chart["mode"] == "video"
    assert chart["value"] == 25.0
    assert chart["title"] == "Gen progress"


def test_minimax_fetch_live_parses_list(monkeypatch):
    from monitor import minimax

    def fake_get(path: str, timeout: float = 0.4):
        if path == "/v1/videos/generations":
            return {
                "object": "list",
                "data": [
                    {
                        "id": "vid_done",
                        "status": "completed",
                        "progress": 1.0,
                        "started_at": 1,
                    },
                    {
                        "id": "vid_run",
                        "status": "running",
                        "phase": "denoising",
                        "progress": 0.42,
                        "step": 3,
                        "total_steps": 8,
                        "started_at": 99,
                        "request": {"mode": "fl2va"},
                    },
                ],
            }
        if path == "/v1/health":
            return {"model": "MiniMaxAI/MiniMax-H3"}
        return None

    monkeypatch.setattr(minimax, "_get_json", fake_get)
    row = minimax.fetch_live()
    assert row is not None
    assert row["kind"] == "video"
    assert row["model"] == "MiniMax-H3"
    assert row["progress"] == 0.42
    assert row["mode"] == "fl2va"
    assert row["job_id"] == "vid_run"


def test_term_render_frame_video():
    from monitor import term

    hist = term.new_history(n=8)
    snap = {
        "card_type": "p150a",
        "architecture": "Blackhole · Tensix",
        "fw_bundle": "18.10.0.0",
        "pcie": "32.0 GT/s x16",
        "kmd": "2.11.0",
        "device_present": True,
        "temp_c": 70.0,
        "power_w": 120.0,
        "aiclk_mhz": 1350.0,
        "ram_used_gib": 8.0,
        "ram_total_gib": 30.0,
    }
    chart = {
        "mode": "video",
        "value": 42.0,
        "model": "MiniMax-H3",
        "step": 3,
        "total_steps": 8,
    }
    frame = term.render_frame(snap, "Denoising  ·  MiniMax-H3  ·  42%", chart, hist)
    assert "TensCUDA Monitor (tty)" in frame
    assert "Gen progress" in frame
    assert "42%" in frame
    assert "Temperature" in frame
    assert "Host RAM" in frame


def test_main_dispatches_tty(monkeypatch):
    import monitor.__main__ as entry
    import monitor.term as term_mod

    called = {}

    def fake_term(argv):
        called["argv"] = list(argv)
        return 42

    monkeypatch.setattr(term_mod, "main", fake_term)
    assert entry._wants_tty(["monitor", "--tty"])
    assert entry._wants_tty(["monitor", "tty", "--interval", "2"])
    assert not entry._wants_tty(["monitor"])
    assert entry.main(["monitor", "--tty", "--interval", "2"]) == 42
    assert called["argv"] == ["--interval", "2"]
