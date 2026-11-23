"""``python -m tenscuda bench``: a fixed, comparable LLM decode benchmark for Tenstorrent cards.

    python -m tenscuda bench                     # default model (smollm2-135m), full run
    python -m tenscuda bench --model tinyllama --quick
    python -m tenscuda bench --list

Fixed workload: one chat prompt, greedy decoding, seed 0, batch 1, bf16 weights loaded with
the streaming loader, a fixed number of new tokens, 1 cold + N warm generate calls. The CPU
parity check runs the same prompt on the CPU in a separate process and compares tokens.

Nothing is uploaded. The result JSON is scrubbed (no hostname, username, paths or IP addresses),
printed in full, and only written to disk after you confirm (or pass --yes). To share it, paste it
into a "Benchmark result" GitHub issue, or open a PR adding it to benchmarks/results/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import glob
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time

SCHEMA = "tenscuda-bench/1"
PROMPT = "Explain in two sentences why the sky is blue."
SEED = 0
MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}
# (new tokens, warm runs)
WORKLOADS = {"full": (32, 3), "quick": (16, 1)}
NEAR_TIE = 0.25  # CPU top-1/top-2 logit margin treated as a bf16-rounding tie


# ----------------------------------------------------------------------------- system info
def _read(path, default=None):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default


def _ver(pkg):
    try:
        import importlib.metadata as md

        return md.version(pkg)
    except Exception:
        return None


def _cards():
    cards = []
    for d in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*")):
        dev = os.path.join(d, "device")
        cards.append({
            "type": _read(os.path.join(d, "tt_card_type")),
            "fw_bundle": _read(os.path.join(d, "tt_fw_bundle_ver")),
            "m3_app_fw": _read(os.path.join(d, "tt_m3app_fw_ver")),
            "aiclk_mhz": _read(os.path.join(d, "tt_aiclk")),
            "pcie": f"{(_read(os.path.join(dev, 'current_link_speed'), '?') or '?').replace(' PCIe', '')} x{_read(os.path.join(dev, 'current_link_width'), '?')}",
        })
    return cards


def _hwmon():
    return sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/device/hwmon/hwmon*"))


def _git_commit():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        c = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=5)
        if c.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no", "--", "tenscuda"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return c.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def system_info():
    cpu = None
    for line in (_read("/proc/cpuinfo", "") or "").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem_kib = int(((_read("/proc/meminfo", "") or "").split("MemTotal:")[1].split()[0])) if os.path.exists("/proc/meminfo") else 0
    osr = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', _read("/etc/os-release", "") or "", re.M))
    import torch

    return {
        "cards": _cards(),
        "card_count": len(_cards()),
        "kmd": _read("/sys/module/tenstorrent/version"),
        "software": {
            "tenscuda": __import__("tenscuda").__version__, "tenscuda_commit": _git_commit(),
            "pjrt_plugin_tt": _ver("pjrt-plugin-tt"), "torch": torch.__version__,
            "torch_xla": _ver("torch-xla"), "tt_lang": _ver("tt-lang"), "transformers": _ver("transformers"),
            "python": platform.python_version(),
        },
        "host": {"cpu": cpu, "cpu_threads": os.cpu_count(), "ram_gib": round(mem_kib / 2**20, 1),
                 "os": osr.get("PRETTY_NAME"), "kernel": platform.release(), "arch": platform.machine()},
    }


# ----------------------------------------------------------------------------- scrubbing
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b")
_MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
_PATH = re.compile(r"(?:/home|/Users|/root|/mnt|/tmp|/var|/opt|/srv|[A-Za-z]:\\\\Users)[^\s\"',;)]*")


_VERSION_KEYS = {"fw_bundle", "m3_app_fw", "kmd"}  # dotted versions like 18.10.0.0, not addresses


def scrub(obj, secrets=None, key=None):
    """Remove hostname, username, absolute paths, IP and MAC addresses from every string."""
    if secrets is None:
        secrets = {s for s in (socket.gethostname(), socket.getfqdn(), _user()) if s and len(s) >= 3}
    if isinstance(obj, dict):
        return {k: scrub(v, secrets, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, secrets, key) for v in obj]
    if isinstance(obj, str):
        s = _PATH.sub("<path>", obj)
        s = _MAC.sub("<mac>", s)
        if key not in _VERSION_KEYS:
            s = _IPV4.sub("<ip>", s)
        s = _IPV6.sub("<ip>", s)
        for sec in sorted(secrets, key=len, reverse=True):
            s = re.sub(r"\b" + re.escape(sec) + r"\b", "<redacted>", s, flags=re.I)
        return s
    return obj


def _user():
    try:
        return getpass.getuser()
    except Exception:
        return None


# ----------------------------------------------------------------------------- telemetry
class Telemetry:
    """Samples card temperature (C) and power (W) from hwmon once a second."""

    def __init__(self):
        self.h = _hwmon()
        self.temps, self.powers = [], []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            for h in self.h:
                t, p = _read(os.path.join(h, "temp1_input")), _read(os.path.join(h, "power1_input"))
                if t:
                    self.temps.append(int(t) / 1000)
                if p:
                    self.powers.append(int(p) / 1e6)
            self._stop.wait(1.0)

    def __enter__(self):
        if self.h:
            self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    def summary(self):
        if not self.temps and not self.powers:
            return None
        return {"temp_c_max": max(self.temps) if self.temps else None,
                "power_w_mean": round(sum(self.powers) / len(self.powers), 1) if self.powers else None,
                "power_w_max": max(self.powers) if self.powers else None, "samples": len(self.temps)}


# ----------------------------------------------------------------------------- benchmark
def _avail_ram():
    try:
        return int(_read("/proc/meminfo").split("MemAvailable:")[1].split()[0]) * 1024
    except Exception:
        return 0


def cpu_reference(model_id, n_params, new_tokens, dtype=None):
    """Run the CPU greedy reference in a child process; cache it under ~/.cache/tenscuda."""
    if dtype is None:
        avail = _avail_ram()
        dtype = "float32" if n_params * 4 < 0.75 * avail else "bfloat16" if n_params * 2 < 0.75 * avail else None
    if dtype is None:
        return None, "skipped: model does not fit in host RAM for a CPU reference"
    cache = os.path.join(os.path.expanduser("~/.cache/tenscuda"), "ref")
    os.makedirs(cache, exist_ok=True)
    key = hashlib.sha1(f"{model_id}|{dtype}|{new_tokens}|{PROMPT}".encode()).hexdigest()[:16]
    out = os.path.join(cache, f"{key}.json")
    if not os.path.exists(out):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cpu_ref.py")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([sys.executable, script, model_id, dtype, str(new_tokens), out, PROMPT],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            return None, "CPU reference failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    return json.load(open(out)), dtype


def run_bench(model_key, quick=False, parity=True):
    import torch

    import tenscuda
    from tenscuda.generate import static_greedy_generate
    from tenscuda.loader import RssMonitor, load_causal_lm, peak_rss_gib
    from transformers import AutoConfig, AutoTokenizer

    if not tenscuda.is_active:
        sys.exit("tenscuda bench: no Tenstorrent device found")
    model_id = MODELS.get(model_key, model_key)
    new_tokens, warm_runs = WORKLOADS["quick" if quick else "full"]
    torch.manual_seed(SEED)
    sysinfo = system_info()

    cfg = AutoConfig.from_pretrained(model_id)
    ref, ref_note = (None, "disabled (--no-parity)")
    if parity:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            skel = AutoModelForCausalLM.from_config(cfg)
        skel.tie_weights()
        n_params = sum(p.numel() for p in skel.parameters())
        del skel
        print(f"[bench] CPU reference for {model_id} ...", flush=True)
        ref, ref_note = cpu_reference(model_id, n_params, new_tokens)

    try:
        from monitor.activity import emit as _activity
    except Exception:
        def _activity(*_a, **_k):
            pass

    n_params_label = f"{n_params / 1e6:.0f}M" if "n_params" in locals() else None

    print(f"[bench] loading {model_id} onto the card (streaming) ...", flush=True)
    _activity("loading", model=model_id, params=n_params_label)
    tok = AutoTokenizer.from_pretrained(model_id)
    mon = RssMonitor()
    t = time.time()
    m = load_causal_lm(model_id, dtype=torch.bfloat16, verbose=False)
    load_s = time.time() - t
    rss_load = peak_rss_gib()
    _activity("decoding", model=model_id, params=n_params_label, peak_rss_gib=round(rss_load, 2))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _cpu_ref import prompt_ids

    ids = prompt_ids(tok, PROMPT)
    runs = []
    with Telemetry() as tel:
        for r in range(1 + warm_runs):
            st = {}
            t = time.time()
            o = static_greedy_generate(m, ids, max_new_tokens=new_tokens, stats=st)
            total = time.time() - t
            dec = st["step_seconds"][1:]
            tok_s = round(len(dec) / sum(dec), 2) if dec and sum(dec) > 0 else 0.0
            runs.append({"total_s": round(total, 2), "first_token_s": round(st["step_seconds"][0], 3),
                         "decode_tok_per_s": tok_s,
                         "tokens": o[0, ids.shape[1]:].cpu().tolist()})
            _activity("decoding", model=model_id, params=n_params_label, tok_s=tok_s, run=r)
            print(f"[bench] run {r} ({'cold' if r == 0 else 'warm'}): {runs[-1]['decode_tok_per_s']} tok/s, "
                  f"{total:.1f} s", flush=True)
    _activity("idle", model=model_id, params=n_params_label,
              tok_s=runs[-1]["decode_tok_per_s"] if runs else None)
    toks = runs[-1]["tokens"]
    par = {"reference": ref_note if ref is None else f"CPU {ref_note}", "status": "skipped"}
    if ref is not None:
        n = next((i for i, (a, b) in enumerate(zip(ref["tokens"], toks)) if a != b), len(ref["tokens"]))
        par.update(matching_prefix=n, total=len(ref["tokens"]))
        if n == len(ref["tokens"]):
            par["status"] = "pass"
        else:
            # Teacher-forced: TT argmax at every position of the CPU sequence.
            seq = torch.cat([ids, torch.tensor([ref["tokens"]])], 1)
            with torch.no_grad():
                lg = m(input_ids=seq.to(tenscuda.device), use_cache=False).logits[0, ids.shape[1] - 1:-1]
                am = lg.float().argmax(-1).cpu().tolist()
            dis = [i for i, (a, b) in enumerate(zip(am, ref["tokens"])) if a != b]
            margins = [ref["margins"][i] for i in dis]
            par.update(teacher_forced_agree=len(am) - len(dis), cpu_margins_at_disagreement=margins)
            par["status"] = "pass (near-tie)" if all(x <= NEAR_TIE for x in margins) else "fail"
    warm = [r["decode_tok_per_s"] for r in runs[1:]]
    try:
        import torch_xla.debug.metrics as met

        cm = met.metric_data("CompileTime")
        fallbacks = sorted(met.executed_fallback_ops())
    except Exception:
        cm, fallbacks = None, []
    res = {
        "schema": SCHEMA,
        "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workload": {"model": model_id, "model_key": model_key if model_key in MODELS else None,
                     "mode": "quick" if quick else "full", "new_tokens": new_tokens, "warm_runs": warm_runs,
                     "prompt": PROMPT, "prompt_tokens": ids.shape[1], "seed": SEED, "decoding": "greedy",
                     "weights": "bf16", "batch": 1},
        "results": {
            "warm_tok_per_s": round(sum(warm) / len(warm), 2), "warm_tok_per_s_runs": warm,
            "first_run_s": runs[0]["total_s"], "first_token_cold_s": runs[0]["first_token_s"],
            "first_token_warm_s": runs[-1]["first_token_s"],
            "xla_compile_s": round(cm[1] / 1e9, 1) if cm else None, "xla_graphs": cm[0] if cm else None,
            "load_s": round(load_s, 1), "peak_host_rss_gib_after_load": round(rss_load, 2),
            "peak_host_rss_gib": mon.stop(), "peak_host_vmhwm_gib": round(peak_rss_gib(), 2), "cpu_fallback_ops": fallbacks,
            "cpu_reference_tok_per_s": ref.get("decode_tok_per_s") if ref else None,
            "parity": par, "telemetry": tel.summary(),
            "runs_identical": all(r["tokens"] == toks for r in runs),
        },
        "config": sysinfo,
    }
    return scrub(res)


def card(res):
    r, w, c = res["results"], res["workload"], res["config"]
    tel = r.get("telemetry") or {}
    cards = c["cards"]
    card_s = f"{c['card_count']}x {cards[0]['type']}" if cards else "none"
    p = r["parity"]
    par = p["status"].upper()
    if "matching_prefix" in p:
        par += f"  ({p['matching_prefix']}/{p['total']} tokens vs {p['reference']}"
        par += f"; teacher-forced {p['teacher_forced_agree']}/{p['total']})" if "teacher_forced_agree" in p else ")"
    lines = [
        ("Model", f"{w['model']}  [{w['mode']}: {w['new_tokens']} tokens, {w['warm_runs']} warm runs]"),
        ("Warm decode", f"{r['warm_tok_per_s']} tok/s  (runs: {', '.join(map(str, r['warm_tok_per_s_runs']))})"),
        ("First run", f"{r['first_run_s']} s  (compile {r['xla_compile_s']} s, {r['xla_graphs']} graphs)"),
        ("First token", f"{r['first_token_cold_s']} s cold / {r['first_token_warm_s']} s warm"),
        ("CPU parity", par),
        ("Peak host RAM", f"{r['peak_host_rss_gib']} GiB anonymous  (VmHWM incl. page cache {r['peak_host_vmhwm_gib']} GiB)"),
        ("Card temp/power", f"max {tel.get('temp_c_max')} C, mean {tel.get('power_w_mean')} W" if tel else "n/a"),
        ("CPU fallback ops", ", ".join(r["cpu_fallback_ops"]) or "none"),
        ("Card", f"{card_s}, fw {cards[0]['fw_bundle'] if cards else '?'}, KMD {c['kmd']}, PCIe {cards[0]['pcie'] if cards else '?'}"),
        ("Software", f"tenscuda {c['software']['tenscuda']} ({c['software']['tenscuda_commit']}), "
                     f"pjrt-plugin-tt {c['software']['pjrt_plugin_tt']}, torch {c['software']['torch']}, "
                     f"torch-xla {c['software']['torch_xla']}"),
        ("Host", f"{c['host']['cpu']}, {c['host']['ram_gib']} GiB RAM, {c['host']['os']}, Python {c['software']['python']}"),
    ]
    width = max(len(k) for k, _ in lines)
    bar = "=" * 72
    return "\n".join([bar, " TensCUDA benchmark".ljust(72), bar] + [f" {k.ljust(width)}  {v}" for k, v in lines] + [bar])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tenscuda bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="smollm2-135m", help="a key from --list, or a Hugging Face model id")
    ap.add_argument("--quick", action="store_true", help="16 tokens, 1 warm run")
    ap.add_argument("--no-parity", action="store_true", help="skip the CPU parity check")
    ap.add_argument("--out", default=None, help="result file (default: tenscuda-bench-<model>-<date>.json)")
    ap.add_argument("--yes", action="store_true", help="write the result without asking")
    ap.add_argument("--list", action="store_true", help="list standard models")
    a = ap.parse_args(argv)
    if a.list:
        for k, v in MODELS.items():
            print(f"{k:14s} {v}")
        return 0
    res = run_bench(a.model, quick=a.quick, parity=not a.no_parity)
    print()
    print(card(res))
    print("\nResult JSON (exactly what would be written; nothing is uploaded):\n")
    print(json.dumps(res, indent=1))
    short = (res["workload"]["model_key"] or res["workload"]["model"].split("/")[-1]).lower()
    out = a.out or f"tenscuda-bench-{short}-{res['date'][:10]}.json"
    if not a.yes:
        if not sys.stdin.isatty():
            print(f"\nNot written (non-interactive). Re-run with --yes to write {out}.")
            return 0
        if input(f"\nWrite this to {out}? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not written.")
            return 0
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    print(f"Wrote {out}. To share it: paste it into a 'Benchmark result' issue, or open a PR adding it "
          f"to benchmarks/results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
