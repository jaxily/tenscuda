"""One command per model: CPU reference + TT run -> results/matrix/<model>.json -> docs/COMPATIBILITY.md.

    python scripts/matrix.py MODEL_ID [--new-tokens 32] [--cpu-dtype auto|float32|bfloat16|skip]
    python scripts/matrix.py MODEL_ID --compile-bench   # explicit compile (default generate uses compile)
    python scripts/matrix.py MODEL_ID --eager-bench     # TENSCUDA_COMPILE=0 (eager TT-XLA loop)
    python scripts/matrix.py --render          # rebuild docs/COMPATIBILITY.md from results/matrix/

Each step runs in its own process, so host RAM is released between them:
  1. CPU reference (scripts/cpu_ref.py): plain greedy loop, fp32 if it fits in RAM, else bf16,
     skipped if neither fits. Records top1-top2 logit margins per step.
  2. TT run (scripts/load_bench.py): streaming loader, 4 generate calls (1 cold + 3 warm).
  3. Status: "works" = all tokens match the CPU reference. "works (near-tie)" = the first
     divergence is at a step where the CPU's top-2 logits are within --tie of each other
     (bf16 rounding can flip the choice), "partial" = diverges elsewhere, "fails" = TT error.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MDIR = os.path.join(ROOT, "results", "matrix")
DOC = os.path.join(ROOT, "docs", "COMPATIBILITY.md")
COLS = ["model", "params", "arch", "status", "parity vs CPU", "warm tok/s (TT)", "first run s",
        "peak host RSS (TT)", "CPU tok/s", "M4 pause Δ%", "notes"]


def _mem_available_gib():
    return int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 2**20


def run(cmd, log, timeout, min_avail_gib=0.6):
    """Run a step in its own process. Kill it if host MemAvailable drops below min_avail_gib
    (keeps the host responsive and out of deep swap); returns "lowmem" in that case."""
    import time

    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.path.join(ROOT, "scripts")})
        t0 = time.time()
        while p.poll() is None:
            if time.time() - t0 > timeout:
                p.kill(); p.wait(); return "timeout"
            if _mem_available_gib() < min_avail_gib:
                p.kill(); p.wait(); return "lowmem"
            time.sleep(0.5)
        return p.returncode


def last_error(log):
    try:
        lines = [l.strip() for l in open(log, errors="replace") if l.strip()]
    except OSError:
        return "no log"
    for l in reversed(lines):
        if any(k in l for k in ("Error", "error:", "Exception", "Killed", "TT_THROW", "FATAL")):
            return l[:220]
    return lines[-1][:220] if lines else "empty log"


def _pause_benchmark(model: str, log: str, timeout: int) -> dict | None:
    """Subprocess A/B replay (parent must not import tenscuda)."""
    import glob as g

    rc = run(
        [PY, "scripts/m4_pause_compare.py", model],
        log,
        timeout,
    )
    if rc != 0:
        return {"error": f"rc={rc}"}
    files = sorted(g.glob(os.path.join(ROOT, "results", "m4_pause_compare_*.json")))
    if not files:
        return {"error": "no json"}
    data = json.load(open(files[-1]))
    row = next((c for c in data.get("comparisons", []) if c.get("model") == model), None)
    if not row:
        return {"error": "model not in compare json"}
    return {
        "baseline_ms": row.get("baseline_mean_decode_ms"),
        "pause_ms": row.get("pause_mean_decode_ms"),
        "delta_pct": row.get("delta_pct_faster"),
    }


def measure(a):
    from transformers import AutoConfig

    short = a.model.split("/")[-1]
    os.makedirs(MDIR, exist_ok=True)
    cfg = AutoConfig.from_pretrained(a.model)
    arch = getattr(cfg, "model_type", "?")
    try:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            skel = AutoModelForCausalLM.from_config(cfg)
        skel.tie_weights()  # count tied embeddings once
        n_params = sum(p.numel() for p in skel.parameters())
        del skel
    except Exception:
        n_params = 0
    avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) * 1024
    cpu_dtype = a.cpu_dtype
    if cpu_dtype == "auto":
        # bf16 checkpoints load memory-mapped (file-backed pages), so bf16 runs even when the
        # model is bigger than free RAM; it's just slow. The low-memory watchdog in run() stops it
        # if anonymous memory actually runs out.
        cpu_dtype = "float32" if n_params * 4 < 0.75 * avail else "bfloat16"
    rec = {"model": a.model, "params": n_params, "arch": arch, "new_tokens": a.new_tokens,
           "cpu_dtype": cpu_dtype, "cpu": None, "tt": None, "tt_error": None}
    if cpu_dtype != "skip":
        out = os.path.join(MDIR, f"{short}_cpu_{cpu_dtype}.json")
        rc = run([PY, "scripts/cpu_ref.py", a.model, cpu_dtype, str(a.new_tokens), out],
                 os.path.join(MDIR, f"{short}_cpu.log"), a.cpu_timeout)
        if rc == 0:
            rec["cpu"] = json.load(open(out))
        else:
            rec["cpu_error"] = f"rc={rc}: {last_error(os.path.join(MDIR, f'{short}_cpu.log'))}"
    tt_json = os.path.join(ROOT, f"results/load_{short}_stream.json")
    if os.path.exists(tt_json):
        os.remove(tt_json)
    log = os.path.join(MDIR, f"{short}_tt.log")
    bench_env = {}
    if a.eager_bench:
        bench_env["TENSCUDA_COMPILE"] = "0"
        rec["notes"] = (rec.get("notes") or "") + " eager-bench"
    elif a.compile_bench:
        bench_env["TENSCUDA_COMPILE"] = "1"
        rec["notes"] = (rec.get("notes") or "") + " compile-bench"
    else:
        bench_env["TENSCUDA_COMPILE"] = "1"
        rec["notes"] = (rec.get("notes") or "") + " compile (default)"
    if rec["cpu"]:
        bench_env["TENSCUDA_TEACHER"] = os.path.join(MDIR, f"{short}_cpu_{cpu_dtype}.json")
    cmd = [PY, "-u", "scripts/load_bench.py", a.model, "stream", str(a.new_tokens)]

    def _run_tt():
        import time
        import subprocess as sp

        with open(log, "w") as fh:
            p = sp.Popen(
                cmd,
                cwd=ROOT,
                stdout=fh,
                stderr=sp.STDOUT,
                env={**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.path.join(ROOT, "scripts"), **bench_env},
            )
            t0 = time.time()
            while p.poll() is None:
                if time.time() - t0 > a.timeout:
                    p.kill()
                    p.wait()
                    return "timeout"
                if _mem_available_gib() < 0.6:
                    p.kill()
                    p.wait()
                    return "lowmem"
                time.sleep(0.5)
            return p.returncode

    rc = _run_tt()
    if bench_env.get("TENSCUDA_TEACHER"):
        del bench_env["TENSCUDA_TEACHER"]
    if rc == 0 and os.path.exists(tt_json):
        rec["tt"] = json.load(open(tt_json))
    else:
        rec["tt_error"] = f"rc={rc}: {last_error(log)}"
    if a.pause_benchmark and rec.get("tt"):
        rec["m4_pause"] = _pause_benchmark(
            a.model, os.path.join(MDIR, f"{short}_pause.log"), a.timeout,
        )
    json.dump(rec, open(os.path.join(MDIR, f"{short}.json"), "w"), indent=1)
    return rec


def row(rec, tie):
    p = rec["params"]
    params = f"{p/1e9:.2f}B" if p >= 1e9 else f"{p/1e6:.0f}M"
    cpu, tt = rec.get("cpu"), rec.get("tt")
    cpu_s = f"{cpu['decode_tok_per_s']} ({rec['cpu_dtype']})" if cpu else "-"
    notes = [rec["notes"]] if rec.get("notes") else []
    if rec.get("cpu_error"):
        notes.append("CPU reference failed: " + rec["cpu_error"])
    if tt is None:
        return [rec["model"], params, rec["arch"], "fails", "-", "-", "-", "-", cpu_s, "-",
                "; ".join(notes + [rec.get("tt_error") or "TT run failed"]).replace("|", "/")]
    warm = [r["decode_tok_per_s"] for r in tt["runs"][1:]]
    toks = tt["runs"][-1]["tokens"]
    all_same = all(r["tokens"] == toks for r in tt["runs"])
    if cpu is None:
        status, parity = "runs (no CPU ref)", "-"
    else:
        ref = cpu["tokens"]
        n = next((i for i, (x, y) in enumerate(zip(ref, toks)) if x != y), len(ref))
        parity = f"{n}/{len(ref)} vs CPU {rec['cpu_dtype']}"
        if n == len(ref):
            status = "works"
        elif cpu["margins"][n] <= tie and (not tt.get("teacher_forced") or
                                           all(x <= tie for x in tt["teacher_forced"]["cpu_margin_at_disagree"])):
            status = "works (near-tie)"
            parity += f"; CPU top-2 margin {cpu['margins'][n]} at step {n}"
        else:
            status = "partial"
            parity += f"; CPU top-2 margin {cpu['margins'][n]} at step {n}"
    tf = tt.get("teacher_forced")
    if tf:
        parity += f"; teacher-forced top-1 {tf['agree']}/{tf['total']}"
        if tf["disagree_steps"]:
            parity += f" (CPU margins {', '.join(str(x) for x in tf['cpu_margin_at_disagree'])})"
    if not all_same:
        notes.append("TT runs not identical")
    notes.append(f"compile {tt['xla_compile_s']} s / {tt['xla_compiles']} graphs")
    notes.append("CPU fallbacks: " + (", ".join(tt["cpu_fallback_ops"]) or "none"))
    pause_col = "-"
    mp = rec.get("m4_pause")
    if mp and mp.get("delta_pct") is not None:
        pause_col = f"{mp['delta_pct']}% ({mp.get('pause_ms')} ms)"
    elif mp and mp.get("error"):
        pause_col = f"err: {mp['error'][:40]}"
    return [rec["model"], params, rec["arch"], status, parity,
            f"{sum(warm)/len(warm):.2f} ({min(warm):.2f}-{max(warm):.2f})", f"{tt['runs'][0]['total_s']:.0f}",
            (f"{tt['peak_anon_rss_gib']} GiB anon (VmHWM {tt['peak_rss_gib_end']})" if "peak_anon_rss_gib" in tt
               else f"{tt['peak_rss_gib_after_load']} load / {tt['peak_rss_gib_end']} end GiB (VmHWM)"), cpu_s,
            pause_col,
            "; ".join(notes).replace("|", "/")]


def render(tie):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(MDIR, "*.json")))
            if "_cpu_" not in os.path.basename(f)]
    recs.sort(key=lambda r: r["params"])
    head = open(os.path.join(ROOT, "docs", "compatibility_header.md")).read().rstrip() + "\n\n"
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    lines += ["| " + " | ".join(row(r, tie)) + " |" for r in recs]
    foot = os.path.join(ROOT, "docs", "compatibility_footer.md")
    tail = "\n\n" + open(foot).read().strip() + "\n" if os.path.exists(foot) else "\n"
    open(DOC, "w").write(head + "\n".join(lines) + tail)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--cpu-dtype", default="auto")
    ap.add_argument("--tie", type=float, default=0.25, help="near-tie threshold on CPU top-2 logit margin")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--cpu-timeout", type=int, default=5400)
    ap.add_argument("--render", action="store_true")
    ap.add_argument(
        "--pause-benchmark",
        action="store_true",
        help="after TT run, subprocess m4_pause_compare for this model (~5 min on P150a)",
    )
    ap.add_argument(
        "--compile-bench",
        action="store_true",
        help="run load_bench with TENSCUDA_COMPILE=1 (explicit; default is already compile)",
    )
    ap.add_argument(
        "--eager-bench",
        action="store_true",
        help="run load_bench with TENSCUDA_COMPILE=0 (slow eager path for A/B)",
    )
    a = ap.parse_args()
    if a.eager_bench and a.compile_bench:
        ap.error("use only one of --eager-bench or --compile-bench")
    if a.model:
        measure(a)
    render(a.tie)


if __name__ == "__main__":
    main()
