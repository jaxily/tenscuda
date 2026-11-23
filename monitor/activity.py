"""Best-effort TensCUDA workload activity log (JSONL). Safe to call without the GUI."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

ACTIVITY_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tenscuda")
ACTIVITY_PATH = os.path.join(ACTIVITY_DIR, "activity.jsonl")
_MAX_BYTES = 512 * 1024
_HEX_REV = re.compile(r"^[0-9a-f]{7,40}$", re.I)
_HF_CACHE = re.compile(r"models--([^/]+)")


def short_model_name(model: Any) -> str | None:
    """Friendly model label; never a HF ``snapshots/<hash>`` folder name."""
    if model is None:
        return None
    s = str(model).rstrip("/")
    if not s:
        return None
    # ~/.cache/huggingface/hub/models--Org--Name/snapshots/<rev>
    m = _HF_CACHE.search(s)
    if m:
        return m.group(1).split("--")[-1]
    last = s.rsplit("/", 1)[-1]
    if _HEX_REV.fullmatch(last):
        return None
    return last


def emit(phase: str, **fields: Any) -> None:
    """Append one activity event. Never raises into the caller."""
    try:
        os.makedirs(ACTIVITY_DIR, exist_ok=True)
        if "model" in fields:
            nice = short_model_name(fields.get("model"))
            if nice:
                fields = {**fields, "model": nice}
            else:
                fields = {k: v for k, v in fields.items() if k != "model"}
        row = {"ts": time.time(), "phase": phase, **{k: v for k, v in fields.items() if v is not None}}
        with open(ACTIVITY_PATH, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        try:
            if os.path.getsize(ACTIVITY_PATH) > _MAX_BYTES:
                _trim()
        except OSError:
            pass
    except Exception:
        pass


def _trim() -> None:
    try:
        with open(ACTIVITY_PATH) as fh:
            lines = fh.readlines()
        keep = lines[-200:]
        with open(ACTIVITY_PATH, "w") as fh:
            fh.writelines(keep)
    except OSError:
        pass


def latest(max_age_s: float = 120.0) -> dict[str, Any] | None:
    """Return the newest activity row if it is recent; else None (treat as idle)."""
    try:
        with open(ACTIVITY_PATH) as fh:
            lines = fh.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = row.get("ts")
        if isinstance(ts, (int, float)) and (time.time() - ts) > max_age_s:
            return None
        return row
    return None


def format_workload(row: dict[str, Any] | None) -> str:
    if not row:
        return "Idle"
    phase = str(row.get("phase") or "running")
    parts = [phase.capitalize()]
    short = short_model_name(row.get("model"))
    if short:
        parts.append(short)
    if row.get("params"):
        parts.append(str(row["params"]))
    if row.get("tok_s") is not None:
        parts.append(f"{row['tok_s']} tok/s")
    if row.get("load_gib") is not None:
        parts.append(f"{row['load_gib']} GiB loaded")
    return "  ·  ".join(parts)


def latest_tok_s(max_age_s: float = 15.0) -> float | None:
    """Rolling decode speed from the newest activity row, or None if idle/stale."""
    info = latest_decode(max_age_s=max_age_s)
    return info["tok_s"] if info else None


def latest_decode(max_age_s: float = 15.0) -> dict[str, Any] | None:
    """Active decode info: ``tok_s``, short ``model`` name. None if idle/stale."""
    row = latest(max_age_s=max_age_s)
    if not row:
        return None
    phase = str(row.get("phase") or "")
    if phase in ("idle", "loading"):
        return None
    tok_s = row.get("tok_s")
    if tok_s is None:
        return None
    try:
        rate = float(tok_s)
    except (TypeError, ValueError):
        return None
    return {
        "tok_s": rate,
        "model": short_model_name(row.get("model")),
        "params": row.get("params"),
    }
