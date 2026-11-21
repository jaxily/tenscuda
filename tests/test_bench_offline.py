"""Benchmark helpers that need no card: scrubbing and leaderboard building."""
import json
import os
import subprocess
import sys

from tenscuda import bench

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_scrub_removes_identity_and_keeps_versions():
    out = bench.scrub({"fw_bundle": "18.10.0.0", "kmd": "2.11.0",
                       "msg": "run by alice on myhost from /home/alice/x.py via 192.168.1.20 fe80:0:0:0:1:2:3:4",
                       "lst": ["10.0.0.5"]}, {"alice", "myhost"})
    assert out["fw_bundle"] == "18.10.0.0" and out["kmd"] == "2.11.0"
    s = json.dumps(out)
    for leak in ("alice", "myhost", "/home", "192.168", "10.0.0.5", "fe80"):
        assert leak not in s, leak


def test_system_info_has_no_host_identity():
    info = bench.scrub(bench.system_info())
    s = json.dumps(info)
    import socket

    assert socket.gethostname() not in s or len(socket.gethostname()) < 3
    assert "/home/" not in s


def test_leaderboard_check_passes_on_repo_results():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_leaderboard.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
