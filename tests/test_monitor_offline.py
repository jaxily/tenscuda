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
