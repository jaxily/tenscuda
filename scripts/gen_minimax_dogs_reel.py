#!/usr/bin/env python3
"""One-shot t2va on the P150 H3 server at the measured max clip length."""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("MINIMAX_URL", "http://127.0.0.1:20000")
# P150 measured ceiling: 124 frames @ 768p ≈ 5.2 s (NFE 4 ~160 s warm).
NUM_FRAMES = int(os.environ.get("H3_NUM_FRAMES", "124"))
STEPS = int(os.environ.get("H3_STEPS", "4"))
SEED = int(os.environ.get("H3_SEED", "1225"))

prompt = (
    "Three cheerful dogs singing Merry Christmas together in fresh snow, "
    "mouths moving in sync with a warm holiday carol, soft Christmas lights "
    "and falling snow, joyful faces looking at camera, vertical 9:16 phone reel, "
    "clear audible dogs singing Merry Christmas with jingle bells."
)
body = json.dumps(
    {
        "prompt": prompt,
        "mode": "t2va",
        "aspect_ratio": "9:16",
        "num_frames": NUM_FRAMES,
        "steps": STEPS,
        "seed": SEED,
    }
).encode()
req = urllib.request.Request(
    f"{BASE}/v1/videos/generations",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as r:
    resp = json.load(r)
print(json.dumps({k: resp[k] for k in ("id", "status", "request") if k in resp}, indent=2))
vid = resp["id"]
out = os.path.expanduser(
    f"~/videos/minimax_h3/reel_dogs_merry_christmas_seed{SEED}_f{NUM_FRAMES}_s{STEPS}.mp4"
)
os.makedirs(os.path.dirname(out), exist_ok=True)
# 124@NFE4 ~3 min; 124@NFE8 can exceed 5 min — poll generously.
for i in range(240):
    with urllib.request.urlopen(f"{BASE}/v1/videos/generations/{vid}", timeout=30) as r:
        st = json.load(r)
    status = st.get("status")
    print(
        f"t={i + 1} status={status} progress={st.get('progress')} step={st.get('step')}"
    )
    if status == "completed":
        print("timings", json.dumps(st.get("timings"), indent=2))
        print("result", json.dumps(st.get("result"), indent=2))
        with urllib.request.urlopen(
            f"{BASE}/v1/videos/generations/{vid}/download", timeout=180
        ) as r:
            open(out, "wb").write(r.read())
        print("wrote", out, os.path.getsize(out))
        sys.exit(0)
    if status in ("failed", "error", "cancelled"):
        print(json.dumps(st, indent=2))
        sys.exit(1)
    time.sleep(5)
print("TIMEOUT")
sys.exit(2)
