"""Best-effort MiniMax-H3 / tt-model video job probe for the card monitor."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:20000"
_ACTIVE = frozenset({"running", "queued", "pending", "processing", "in_progress"})


def base_url() -> str:
    return (os.environ.get("MINIMAX_URL") or DEFAULT_BASE).rstrip("/")


def _get_json(path: str, timeout: float = 0.4) -> Any | None:
    try:
        req = urllib.request.Request(f"{base_url()}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def short_model(name: Any) -> str | None:
    if not name:
        return None
    s = str(name).rstrip("/")
    if not s:
        return None
    return s.rsplit("/", 1)[-1]


def fetch_live() -> dict[str, Any] | None:
    """Return a monitor activity row for the newest active video job, or None."""
    payload = _get_json("/v1/videos/generations")
    if not isinstance(payload, dict):
        return None
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        return None

    active: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").lower()
        if status in _ACTIVE:
            active.append(row)
    if not active:
        return None

    # Prefer actively running over queued; then most recently started/created.
    def _rank(row: dict[str, Any]) -> tuple:
        status = str(row.get("status") or "").lower()
        running = 0 if status == "running" else 1
        started = row.get("started_at") or row.get("created_at") or 0
        try:
            started_f = float(started)
        except (TypeError, ValueError):
            started_f = 0.0
        return (running, -started_f)

    job = sorted(active, key=_rank)[0]
    req = job.get("request") if isinstance(job.get("request"), dict) else {}
    health = _get_json("/v1/health")
    model = None
    if isinstance(health, dict):
        model = short_model(health.get("model"))
    if not model:
        model = "MiniMax-H3"

    progress = job.get("progress")
    try:
        progress_f = float(progress) if progress is not None else None
    except (TypeError, ValueError):
        progress_f = None

    phase = str(job.get("phase") or job.get("status") or "generating")
    return {
        "ts": job.get("started_at") or job.get("created_at"),
        "phase": phase,
        "kind": "video",
        "model": model,
        "progress": progress_f,
        "step": job.get("step"),
        "total_steps": job.get("total_steps"),
        "mode": req.get("mode") or job.get("mode"),
        "status": job.get("status"),
        "job_id": job.get("id"),
        "elapsed_s": job.get("elapsed_s"),
    }
