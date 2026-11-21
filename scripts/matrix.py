"""One command per model: run CPU reference + TT, append a row to docs/COMPATIBILITY.md.

    python scripts/matrix.py MODEL_ID [--new-tokens 32] [--cpu-dtype float32|bfloat16|skip]

Steps (separate processes so host RAM is released between them):
  1. CPU reference with examples/tinyllama_generate.py (skipped if the model can't fit in RAM).
  2. TT run with scripts/load_bench.py (streaming loader, 4 runs: 1 cold + 3 warm).
  3. Row appended under the MATRIX-ROWS marker. Status is "works" if tokens match,
     "partial" if it runs but diverges, "fails" if the TT run errors.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONPATH": ROOT})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--cpu-dtype", default="auto")
    a = ap.parse_args()
    short = a.model.split("/")[-1]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(a.model)
    arch = getattr(cfg, "model_type", "?")
    try:
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            n_params = sum(p.numel() for p in AutoModelForCausalLM.from_config(cfg).parameters())
    except Exception:
        n_params = 0
    avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    cpu_dtype = a.cpu_dtype
    if cpu_dtype == "auto":
        cpu_dtype = "float32" if n_params * 4 < 0.7 * avail else ("bfloat16" if n_params * 2 < 0.7 * avail else "skip")

    ref = None
    if cpu_dtype != "skip":
        ref_json = f"results/matrix_{short}_cpu_{cpu_dtype}.json"
        rc = run([PY, "examples/tinyllama_generate.py", "--model", a.model, "--dtype", cpu_dtype,
                  "--max-new-tokens", str(a.new_tokens), "--runs", "1", "--out", ref_json],
                 f"results/matrix_{short}_cpu.log")
        if rc == 0:
            ref = json.load(open(os.path.join(ROOT, ref_json)))["runs"][-1]["tokens"]

    rc = run([PY, "-u", "scripts/load_bench.py", a.model, "stream", str(a.new_tokens)],
             f"results/matrix_{short}_tt.log")
    tt_json = os.path.join(ROOT, f"results/load_{short}_stream.json")
    if rc != 0 or not os.path.exists(tt_json):
        row = [a.model, f"{n_params/1e9:.2f}B", arch, "fails", "-", "-", "-", "-", "n/a",
               f"TT run failed (rc={rc}); see results/matrix_{short}_tt.log"]
    else:
        d = json.load(open(tt_json))
        warm = [r["decode_tok_per_s"] for r in d["runs"][1:]]
        toks = d["runs"][-1]["tokens"]
        if ref is None:
            parity, status = "no CPU ref (RAM)", "works?"
        else:
            n = next((i for i, (x, y) in enumerate(zip(ref, toks)) if x != y), len(ref))
            parity = f"{n}/{len(ref)} vs CPU {cpu_dtype}"
            status = "works" if n == len(ref) else "partial"
        row = [a.model, f"{n_params/1e9:.2f}B", arch, status, parity,
               f"{sum(warm)/len(warm):.2f} ({min(warm):.2f}-{max(warm):.2f})", f"{d['runs'][0]['total_s']:.0f}",
               f"{d['peak_rss_gib_after_load']} GiB load / {d['peak_rss_gib_end']} GiB end", "n/a",
               f"compile {d['xla_compile_s']} s, {d['xla_compiles']} graphs; fallbacks {d['cpu_fallback_ops'] or 'none'}"]
    line = "| " + " | ".join(row) + " |"
    path = os.path.join(ROOT, "docs/COMPATIBILITY.md")
    s = open(path).read()
    s = s.replace("<!-- MATRIX-ROWS -->", line + "\n<!-- MATRIX-ROWS -->")
    open(path, "w").write(s)
    print(line)


if __name__ == "__main__":
    main()
