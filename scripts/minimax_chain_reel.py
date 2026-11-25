#!/usr/bin/env python3
"""Chain MiniMax-H3 clips with fl2va last→first frame continuity, then stitch.

P150 max continuous clip is 124 frames (~5.2 s). Default here uses that ceiling;
five scenes ≈ 26 s. Override with --frames 73 for the old 3 s / ~15 s reel.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("MINIMAX_URL", "http://127.0.0.1:20000")
# Measured P150 ceiling in jashansinghTT/minimax-h3-blackhole (NFE 4 @ 768p).
DEFAULT_FRAMES = int(os.environ.get("H3_NUM_FRAMES", "124"))
DEFAULT_STEPS = int(os.environ.get("H3_STEPS", "4"))

_DOGS = (
    "Same three dogs throughout, consistent faces and costumes: "
    "Buddy a golden retriever with a red knit scarf, "
    "Pip a black-and-white border collie with a green bow on her collar, "
    "Snowball a fluffy white samoyed wearing a tiny Santa hat. "
    "Same Christmas house: front door, hallway leading to living room, "
    "Christmas tree on the left, warm fairy lights, wrapping-paper mess on the floor, "
    "snowy window, handheld phone-reel energy, vertical 9:16."
)

CHARACTER_BIBLE_ACTION = (
    f"{_DOGS} Dynamic action, NO singing."
)
CHARACTER_BIBLE_SING = (
    f"{_DOGS} Dogs clearly SING the carol Jingle Bells with audible merry vocals and "
    "jingle bells; mouths move in sync with the lyrics while they run and play."
)
CHARACTER_BIBLE_SPECTACLE = (
    f"{_DOGS} Premium cinematic Christmas action reel — stylized live-action, rich practical light, "
    "anamorphic vertical phone energy, bold camera, crisp diegetic SFX. "
    "NO lip-sync karaoke, NO dogs singing lyrics, NO cartoon rubber faces, NO Pixar CGI look."
)

PRESETS: dict[str, tuple[str, list[dict]]] = {
    "action": (
        CHARACTER_BIBLE_ACTION,
        [
            {
                "id": "01_bolt",
                "seed": 2601,
                "prompt": (
                    f"{CHARACTER_BIBLE_ACTION} "
                    "ACTION HOOK: Front door flies open hard. Buddy BOLTS into frame at full sprint "
                    "from the doorway into the hallway, paws scrambling, scarf flying, "
                    "camera whip-pans to follow him, motion blur, whoosh + claw-click SFX, no singing."
                ),
            },
            {
                "id": "02_chase",
                "seed": 2602,
                "last_keyframe_prompt": (
                    f"{CHARACTER_BIBLE_ACTION} "
                    "End pose: Buddy and Pip stop beside the Christmas tree, wrapping paper at paws, "
                    "looking forward, no singing, clear living-room tree background."
                ),
                "prompt": (
                    f"{CHARACTER_BIBLE_ACTION} "
                    "CHASE: Pip sprints after Buddy down the hallway toward the living room. "
                    "Start near the open door; END near the Christmas tree. "
                    "Low tracking camera, fast run cycle, rugs sliding, no singing."
                ),
            },
            {
                "id": "03_skid",
                "seed": 2603,
                "prompt": (
                    f"{CHARACTER_BIBLE_ACTION} "
                    "SKID: Buddy and Pip slam-brake into wrapping paper under the tree, "
                    "paper explodes, comic skid, rustling SFX, no singing."
                ),
            },
            {
                "id": "04_leap",
                "seed": 2604,
                "prompt": (
                    f"{CHARACTER_BIBLE_ACTION} "
                    "LEAP: Snowball leaps OVER them with a squeaky toy, Santa hat on, "
                    "slow-mo jump arc, camera tilts up, soft landing, no singing."
                ),
            },
            {
                "id": "05_tumble",
                "seed": 2605,
                "prompt": (
                    f"{CHARACTER_BIBLE_ACTION} "
                    "PAYOFF: All three tumble in wrapping paper then pop up looking at camera, "
                    "tongues out, tails wagging, paper falling, viral ending, no singing."
                ),
            },
        ],
    ),
    "singing_action": (
        CHARACTER_BIBLE_SING,
        [
            {
                "id": "01_bolt",
                "seed": 2701,
                "prompt": (
                    f"{CHARACTER_BIBLE_SING} "
                    "HOOK: Front door flies open. Buddy BOLTS in singing Jingle Bells, mouthing "
                    "'Dashing through the snow', scarf flying, whip-pan follow, "
                    "clear carol vocals + jingle bells over paw-scramble SFX."
                ),
            },
            {
                "id": "02_chase",
                "seed": 2702,
                "last_keyframe_prompt": (
                    f"{CHARACTER_BIBLE_SING} "
                    "End pose: Buddy and Pip at the Christmas tree still singing, mouths open mid-lyric, "
                    "wrapping paper at paws, living-room tree behind them."
                ),
                "prompt": (
                    f"{CHARACTER_BIBLE_SING} "
                    "CHASE: Pip chases Buddy down the hall toward the tree while both sing "
                    "'Jingle bells, jingle bells' with mouths in sync, "
                    "low tracking camera, run cycle + merry duet, start at door-hall, END at tree."
                ),
            },
            {
                "id": "03_skid",
                "seed": 2703,
                "prompt": (
                    f"{CHARACTER_BIBLE_SING} "
                    "SKID: Buddy and Pip skid into wrapping paper belting "
                    "'oh what fun it is to ride', paper explodes, loud chorus + rustle SFX."
                ),
            },
            {
                "id": "04_leap",
                "seed": 2704,
                "prompt": (
                    f"{CHARACTER_BIBLE_SING} "
                    "LEAP: Snowball leaps over them with a toy, singing 'jingle all the way' mid-air, "
                    "Santa hat on, slow-mo jump, trio vocals swelling, camera tilts up."
                ),
            },
            {
                "id": "05_tumble",
                "seed": 2705,
                "prompt": (
                    f"{CHARACTER_BIBLE_SING} "
                    "PAYOFF: All three tumble then pop up for the final 'hey!' of Jingle Bells, "
                    "look at camera, mouths open, tails wagging, paper falling, viral ending."
                ),
            },
        ],
    ),
    # Lean into Turbo strengths: bold camera, diegetic SFX, stylized continuity — not photo likeness.
    "spectacle": (
        CHARACTER_BIBLE_SPECTACLE,
        [
            {
                "id": "01_threshold",
                "seed": 3101,
                "prompt": (
                    f"{CHARACTER_BIBLE_SPECTACLE} "
                    "[Shot 1] Cinematic live-action anamorphic vertical phone reel. "
                    "Night exterior: snowing hard outside a warm Christmas house. "
                    "Door SLAMS open toward camera — warm interior light floods the lens. "
                    "Buddy the golden fills the frame mid-sprint, red scarf whipping, "
                    "snow crystals streaking past; whip-pan as he blasts past camera into the hall. "
                    "overall_soundscape: door slam whoosh, wind, paw scramble on wood, distant jingle bells. "
                    "non_diegetic_music: short brass hit + rising sleigh-bell stinger."
                ),
            },
            {
                "id": "02_pursuit",
                "seed": 3102,
                "last_keyframe_prompt": (
                    f"{CHARACTER_BIBLE_SPECTACLE} "
                    "End pose freeze-frame energy: Buddy and Pip skidding to a halt beside the lit "
                    "Christmas tree, wrapping paper at paws, fairy lights bokeh, looking forward, "
                    "scarves/bow still, living-room depth sharp."
                ),
                "prompt": (
                    f"{CHARACTER_BIBLE_SPECTACLE} "
                    "[Shot 1] Low tracking dolly down the hallway — Buddy sprints ahead, "
                    "Pip the border collie rockets after him, green bow bouncing, "
                    "rugs sliding, framed photos rattling on walls. "
                    "[Shot 2] Camera races into the living room and lands beside the Christmas tree. "
                    "overall_soundscape: claw clicks, collar tags, wood creak, muffled carol from a stereo. "
                    "non_diegetic_music: driving percussion under jingle bells, no vocals."
                ),
            },
            {
                "id": "03_detonation",
                "seed": 3103,
                "prompt": (
                    f"{CHARACTER_BIBLE_SPECTACLE} "
                    "[Shot 1] Buddy and Pip slam-brake into a mountain of wrapping paper under the tree; "
                    "paper EXPLODES in slow-mo shards of red/gold/green, fairy lights strobe through debris, "
                    "comic but cinematic — 48fps feel then snap back. "
                    "Camera orbits 90° around them as paper rains. "
                    "overall_soundscape: paper roar, skid squeak, soft cartoon boing under real physics. "
                    "non_diegetic_music: big drum hit then quiet shimmer."
                ),
            },
            {
                "id": "04_airborne",
                "seed": 3104,
                "prompt": (
                    f"{CHARACTER_BIBLE_SPECTACLE} "
                    "[Shot 1] Hero slow-mo: Snowball the samoyed LEAPS over Buddy and Pip with a squeaky toy "
                    "clamped in her jaws, tiny Santa hat staying on, fluffy coat backlit by tree lights "
                    "into a rim-light halo; camera tilts up under the arc then soft-lands with her. "
                    "overall_soundscape: air whoosh, toy squeak, soft thud landing, audience-cheer room tone. "
                    "non_diegetic_music: swelling choir pad + single sleigh-bell cascade — no lyric singing."
                ),
            },
            {
                "id": "05_tableau",
                "seed": 3105,
                "prompt": (
                    f"{CHARACTER_BIBLE_SPECTACLE} "
                    "[Shot 1] Payoff tableau: all three dogs pop up from wrapping paper side-by-side facing camera, "
                    "tongues out, tails a blur, paper still falling like confetti, tree sparkling behind. "
                    "Slow push-in to medium shot, handheld micro-shake, fairy-light bokeh. "
                    "They hold the look — victorious, mischievous, viral ending freeze energy. "
                    "overall_soundscape: happy pants, paper flutter, collar jingles, one soft bark. "
                    "non_diegetic_music: final triumphant jingle-bell sting and cut."
                ),
            },
        ],
    ),
}

# Real two-dog reel: photo keyframes. Keep look photoreal; audio natural (no synth karaoke).
_OUR_DOGS = (
    "Photoreal handheld iPhone video of exactly these two real household dogs only "
    "(match the reference photos — real fur texture, real eyes, natural proportions; "
    "NO cartoon, NO CGI, NO Pixar, NO 3D render, NO airbrushed plastic fur, NO beauty filter): "
    "GOLDEN — medium/large adult golden-apricot dog, thick wavy coat with individual strands, "
    "black nose, dark soulful eyes, long floppy ears. "
    "CREAM — smaller fluffy cream/apricot doodle-mix, curly soft fur, round face, "
    "dark eyes, floppy ears, slight underbite lip. "
    "Same real Christmas house: front door, hallway to living room, tree on the left, "
    "warm fairy lights, wrapping paper on the floor, vertical 9:16 phone reel, mild grain. "
    "Audio: natural room tone — paw clicks, collar tags, paper rustle, happy panting/play barks; "
    "a quiet real stereo in another room plays a soft acoustic Jingle Bells faintly "
    "(human recording, imperfect, distant, muffled by walls — NOT studio Auto-Tune, "
    "NOT TTS, NOT synthetic choir, NOT dogs singing karaoke, NO lip-sync mouthing lyrics)."
)


def _our_dogs_scenes(refs: Path) -> list[dict]:
    golden = str(refs / "dog_golden.jpg")
    cream = str(refs / "dog_cream.jpg")
    return [
        {
            "id": "01_bolt",
            "seed": 2901,
            "first_frame_path": golden,
            "prompt": (
                f"{_OUR_DOGS} "
                "HOOK from photo likeness: Front door flies open. The GOLDEN dog bolts into the "
                "hallway at a real run — coat and ears flying, paws scrambling, natural pant, "
                "whip-pan follow, motion blur. Mouth closed or panting, not singing."
            ),
        },
        {
            "id": "02_chase",
            "seed": 2902,
            "last_keyframe_prompt": (
                f"{_OUR_DOGS} "
                "End pose near the Christmas tree: GOLDEN and CREAM side by side (not stacked), "
                "natural playful faces, wrapping paper at their paws, living-room tree behind them. "
                "Photoreal, mouths closed or soft pant — no exaggerated open-mouth singing pose."
            ),
            "last_keyframe_first_frame_path": cream,
            "prompt": (
                f"{_OUR_DOGS} "
                "CHASE: The little CREAM fluffy dog races after the GOLDEN down the hall to the tree. "
                "Low tracking camera, real run cycles, rugs sliding. Start at door-hall, END at tree. "
                "No singing, no lip-sync."
            ),
        },
        {
            "id": "03_skid",
            "seed": 2903,
            "prompt": (
                f"{_OUR_DOGS} "
                "SKID: Both dogs slam-brake into wrapping paper under the tree; paper explodes, "
                "comic but photoreal skid, rustling SFX, happy play-bark. No singing."
            ),
        },
        {
            "id": "04_leap",
            "seed": 2904,
            "first_frame_path": cream,
            "prompt": (
                f"{_OUR_DOGS} "
                "LEAP: The small CREAM dog leaps playfully over the GOLDEN — real dog physics, "
                "soft fur in motion, brief slow-mo, camera tilts up, soft landing. No singing."
            ),
        },
        {
            "id": "05_tumble",
            "seed": 2905,
            "prompt": (
                f"{_OUR_DOGS} "
                "PAYOFF: Both tumble in wrapping paper then sit/stand side by side looking toward "
                "camera with natural happy dog faces (soft pant or closed mouth — not wide cartoon "
                "grins), tails wagging, paper falling. Photoreal viral ending, no singing."
            ),
        },
    ]


def build_preset(name: str, refs: Path) -> tuple[str, list[dict]]:
    if name == "our_dogs_sing":
        return _OUR_DOGS, _our_dogs_scenes(refs)
    return PRESETS[name]


def api(method: str, path: str, body: dict | None = None, timeout: int = 120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if "json" in (r.headers.get("Content-Type") or ""):
            return json.load(r)
        return r.read()


def _emit_monitor(phase: str, **fields) -> None:
    try:
        from monitor.activity import emit

        emit(phase, kind="video", model="MiniMax-H3", **fields)
    except Exception:
        pass


def wait_job(vid: str, poll_s: float = 5.0, max_polls: int = 360) -> dict:
    for i in range(max_polls):
        st = api("GET", f"/v1/videos/generations/{vid}")
        status = st.get("status")
        print(
            f"  [{i + 1}] {status} progress={st.get('progress')} step={st.get('step')}",
            flush=True,
        )
        _emit_monitor(
            str(st.get("phase") or status or "generating"),
            progress=st.get("progress"),
            step=st.get("step"),
            total_steps=st.get("total_steps"),
            status=status,
            job_id=vid,
            mode=(st.get("request") or {}).get("mode") if isinstance(st.get("request"), dict) else None,
        )
        if status == "completed":
            _emit_monitor("idle")
            return st
        if status in ("failed", "error", "cancelled"):
            _emit_monitor("idle")
            raise RuntimeError(json.dumps(st, indent=2))
        time.sleep(poll_s)
    _emit_monitor("idle")
    raise TimeoutError(vid)


def generate(
    prompt: str,
    seed: int,
    out_mp4: Path,
    *,
    num_frames: int,
    steps: int,
    first_frame_b64: str | None = None,
    last_frame_b64: str | None = None,
) -> dict:
    body: dict = {
        "prompt": prompt,
        "aspect_ratio": "9:16",
        "num_frames": num_frames,
        "steps": steps,
        "seed": seed,
    }
    images = []
    if first_frame_b64:
        images.append(
            {"image": f"data:image/jpeg;base64,{first_frame_b64}", "position": "first"}
        )
    if last_frame_b64:
        images.append(
            {"image": f"data:image/jpeg;base64,{last_frame_b64}", "position": "last"}
        )
    if images:
        body["mode"] = "fl2va"
        body["image_prompts"] = images
    else:
        body["mode"] = "t2va"

    print(
        f"POST {body['mode']} frames={num_frames} steps={steps} seed={seed} "
        f"first={bool(first_frame_b64)} last={bool(last_frame_b64)} → {out_mp4.name}",
        flush=True,
    )
    try:
        resp = api("POST", "/v1/videos/generations", body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e

    vid = resp["id"]
    print(f"  queued {vid}", flush=True)
    st = wait_job(vid)
    raw = api("GET", f"/v1/videos/generations/{vid}/download", timeout=180)
    assert isinstance(raw, (bytes, bytearray))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_mp4.write_bytes(raw)
    meta = {
        "id": vid,
        "elapsed_s": st.get("elapsed_s"),
        "timings": st.get("timings"),
        "result": st.get("result"),
        "path": str(out_mp4),
    }
    out_mp4.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_mp4} ({len(raw)} bytes, {st.get('elapsed_s')}s)", flush=True)
    return meta


def file_jpeg_b64(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def frame_jpeg_b64(mp4: Path, ffmpeg: str, which: str = "last") -> str:
    jpg = mp4.with_name(mp4.stem + f".{which}.jpg")
    if which == "last":
        cmd = [
            ffmpeg,
            "-y",
            "-sseof",
            "-0.05",
            "-i",
            str(mp4),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(jpg),
        ]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            "0",
            "-i",
            str(mp4),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(jpg),
        ]
    subprocess.run(cmd, check=True, capture_output=True)
    return base64.b64encode(jpg.read_bytes()).decode("ascii")


def stitch(clips: list[Path], out: Path, ffmpeg: str) -> None:
    lst = out.with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    print(f"stitched → {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=os.path.expanduser("~/videos/minimax_h3/action_dogs"),
    )
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    ap.add_argument("--start-from", type=int, default=0, help="Resume at scene index")
    ap.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help="Frames per clip (P150 max 124 ≈ 5.2s; 73 ≈ 3.0s)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Denoise NFE (4 = served Turbo; 8 needs H3_QUALITY=8 on the server)",
    )
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS) + ["our_dogs_sing"],
        default="action",
        help="Story arc: action | singing_action | spectacle | our_dogs_sing (photo refs)",
    )
    ap.add_argument(
        "--refs",
        default=os.path.expanduser("~/videos/minimax_h3/refs"),
        help="Directory with dog_golden.jpg and dog_cream.jpg for our_dogs_sing",
    )
    ap.add_argument(
        "--final-name",
        default="",
        help="Output filename (default derived from preset/frames)",
    )
    args = ap.parse_args()
    character_bible, scenes = build_preset(args.preset, Path(args.refs))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    approx_s = args.frames / 24.0
    reel_s = approx_s * len(scenes)
    final_name = args.final_name or f"reel_{args.preset}_{reel_s:.0f}s.mp4"

    health = api("GET", "/v1/health")
    if not health.get("ready"):
        print("server not ready:", health, file=sys.stderr)
        return 1
    print(
        f"ready: {health.get('model')} {health.get('profile')} | preset={args.preset} | "
        f"{args.frames} frames/clip (~{approx_s:.1f}s) × {len(scenes)} ≈ {reel_s:.0f}s | "
        f"steps={args.steps}",
        flush=True,
    )

    clips: list[Path] = []
    prev_b64: str | None = None
    for i, scene in enumerate(scenes):
        mp4 = out_dir / f"{scene['id']}.mp4"
        if i < args.start_from and mp4.exists():
            print(f"skip existing {mp4.name}", flush=True)
            clips.append(mp4)
            prev_b64 = frame_jpeg_b64(mp4, args.ffmpeg, "last")
            continue

        first = prev_b64 if i > 0 else None
        if scene.get("first_frame_path"):
            # Photo lock overrides chain for this beat (identity > motion continuity).
            first = file_jpeg_b64(scene["first_frame_path"])
        last = None
        dest_prompt = scene.get("last_keyframe_prompt")
        if dest_prompt:
            dest_mp4 = out_dir / f"{scene['id']}_endkey.mp4"
            if not (i < args.start_from and dest_mp4.exists()):
                dest_first = None
                if scene.get("last_keyframe_first_frame_path"):
                    dest_first = file_jpeg_b64(scene["last_keyframe_first_frame_path"])
                generate(
                    dest_prompt,
                    scene["seed"] + 1000,
                    dest_mp4,
                    num_frames=args.frames,
                    steps=args.steps,
                    first_frame_b64=dest_first,
                )
            last = frame_jpeg_b64(dest_mp4, args.ffmpeg, "last")

        generate(
            scene["prompt"],
            scene["seed"],
            mp4,
            num_frames=args.frames,
            steps=args.steps,
            first_frame_b64=first,
            last_frame_b64=last,
        )
        clips.append(mp4)
        prev_b64 = frame_jpeg_b64(mp4, args.ffmpeg, "last")

    final = out_dir / final_name
    stitch(clips, final, args.ffmpeg)
    (out_dir / "arc.json").write_text(
        json.dumps(
            {
                "preset": args.preset,
                "character_bible": character_bible,
                "scenes": scenes,
                "frames": args.frames,
                "steps": args.steps,
                "clips": [str(c) for c in clips],
                "final": str(final),
            },
            indent=2,
        )
    )
    print("DONE", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
