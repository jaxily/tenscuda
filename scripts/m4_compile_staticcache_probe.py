#!/usr/bin/env python3
"""M4 research: re-test torch.compile(backend="tt") with upstream StaticCache pattern.

Our PERF §5 result (wrong from token 1) used the old path:
  StaticCache(...) with no device=, then torch.compile(model) + cache_position.

Upstream tt-xla llama tests do this instead:
  1) StaticCache(..., device="cpu", dtype=bf16)
  2) move keys/values/cumulative_length to XLA
  3) model.compile / torch.compile(backend="tt")  # experimental = extract_compiled_graph
  4) one sync per forward

Also A/B legacy compile (mutation-aware sync path in tt_torch backend).

    PYTHONPATH=~/tenscuda .venv/bin/python scripts/m4_compile_staticcache_probe.py \\
        HuggingFaceTB/SmolLM2-135M-Instruct 16

One job at a time on the card host. Writes results/m4_compile_staticcache_<slug>.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.split("/")[-1])


def _prompt_ids(tok):
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    if tok.chat_template:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    return tok(msgs[0]["content"], return_tensors="pt").input_ids


def _xla_compiles() -> int:
    import torch_xla.debug.metrics as met

    row = met.metric_data("CompileTime")
    return int(row[0]) if row else 0


def _cache_head_meta(model):
    cfg = model.config
    # GQA: key/value heads may differ from query heads.
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return n_heads, head_dim


def _early_init_cache(cache, model, *, device, dtype):
    """Transformers ≥5.5 leaves keys/values None until early_initialization / first forward."""
    if not hasattr(cache, "early_initialization"):
        return
    n_heads, head_dim = _cache_head_meta(model)
    cache.early_initialization(
        batch_size=1, num_heads=n_heads, head_dim=head_dim, dtype=dtype, device=device,
    )


def _move_static_cache_to_device(cache, device):
    """Match tt-xla test_llama_step_n300: CPU-allocated buffers, then .to(device)."""
    for layer in cache.layers:
        if hasattr(layer, "keys") and layer.keys is not None:
            layer.keys = layer.keys.to(device)
        if hasattr(layer, "values") and layer.values is not None:
            layer.values = layer.values.to(device)
        if hasattr(layer, "cumulative_length") and layer.cumulative_length is not None:
            cl = layer.cumulative_length
            if hasattr(cl, "to"):
                layer.cumulative_length = cl.to(device)
        if hasattr(layer, "device"):
            layer.device = device


def _make_cache(model, max_len, device, *, cpu_init: bool):
    import torch
    from transformers import StaticCache

    kwargs = {"config": model.config, "max_cache_len": max_len}
    # Transformers versions differ: some accept device/dtype/max_batch_size.
    if cpu_init:
        try:
            cache = StaticCache(**kwargs, device="cpu", dtype=torch.bfloat16, max_batch_size=1)
        except TypeError:
            try:
                cache = StaticCache(**kwargs, device="cpu", dtype=torch.bfloat16)
            except TypeError:
                cache = StaticCache(**kwargs)
        _early_init_cache(cache, model, device="cpu", dtype=torch.bfloat16)
        _move_static_cache_to_device(cache, device)
    else:
        cache = StaticCache(**kwargs)
    return cache


def _greedy_loop(model, ids_cpu, n_new, *, use_compile: bool, cpu_init_cache: bool,
                 legacy_compile: bool, use_position_ids: bool, pad_to: int = 64):
    """Return (tokens, mean_decode_ms, compiles_delta, wall_s, note)."""
    import torch
    import torch_xla
    from tenscuda._patch import real_torch_device

    dev = next(model.parameters()).device
    plen = ids_cpu.shape[1]
    max_len = -(-(plen + n_new) // pad_to) * pad_to
    cache = _make_cache(model, max_len, dev, cpu_init=cpu_init_cache)

    fwd = model
    if use_compile:
        opts = {"tt_legacy_compile": True} if legacy_compile else None
        with real_torch_device():
            compiled = torch.compile(model, backend="tt", dynamic=False, options=opts)

        def fwd(**kwargs):
            with real_torch_device():
                return compiled(**kwargs)

    out_tokens: list[int] = []
    step_ms: list[float] = []
    comp0 = _xla_compiles()
    t_wall = time.time()
    with torch.no_grad():
        # Prefill
        t0 = time.time()
        if use_position_ids:
            pos = torch.arange(plen, device=dev).unsqueeze(0)
            logits = fwd(
                input_ids=ids_cpu.to(dev),
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            ).logits
        else:
            pos = torch.arange(plen, device=dev)
            logits = fwd(
                input_ids=ids_cpu.to(dev),
                cache_position=pos,
                past_key_values=cache,
                use_cache=True,
            ).logits
        torch_xla.sync()
        nxt = int(logits[:, -1, :].argmax(-1).cpu())
        step_ms.append((time.time() - t0) * 1000)
        out_tokens.append(nxt)
        cur = plen

        while len(out_tokens) < n_new:
            t0 = time.time()
            step_ids = torch.tensor([[nxt]], device=dev)
            if use_position_ids:
                step_pos = torch.tensor([[cur]], device=dev)
                logits = fwd(
                    input_ids=step_ids,
                    position_ids=step_pos,
                    past_key_values=cache,
                    use_cache=True,
                ).logits
            else:
                step_pos = torch.tensor([cur], device=dev)
                logits = fwd(
                    input_ids=step_ids,
                    cache_position=step_pos,
                    past_key_values=cache,
                    use_cache=True,
                ).logits
            torch_xla.sync()
            nxt = int(logits[:, -1, :].argmax(-1).cpu())
            step_ms.append((time.time() - t0) * 1000)
            out_tokens.append(nxt)
            cur += 1

    wall = time.time() - t_wall
    dec = step_ms[1:]
    mean_dec = round(sum(dec) / max(len(dec), 1), 1)
    return {
        "tokens": out_tokens,
        "mean_decode_ms": mean_dec,
        "prefill_ms": round(step_ms[0], 1) if step_ms else None,
        "xla_compile_delta": _xla_compiles() - comp0,
        "wall_s": round(wall, 2),
    }


def _run_variant(label, model_id, n_new, **kwargs):
    """Fresh process semantics: caller should invoke one variant per process.
    Within one process we still reload model between variants when possible."""
    import tenscuda  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tenscuda.generate import static_greedy_generate

    if not tenscuda.is_active:
        raise SystemExit("tenscuda: no TT device")

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = _prompt_ids(tok)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()

    if label == "baseline_static_greedy":
        stats: dict = {}
        t0 = time.time()
        out = static_greedy_generate(
            m, ids, max_new_tokens=n_new, eos_token_id=None, stats=stats,
        )
        steps = stats.get("step_seconds", [])
        dec = steps[1:] if len(steps) > 1 else []
        tokens = out[0, ids.shape[1]:].detach().cpu().tolist()
        return {
            "label": label,
            "tokens": [int(t) for t in tokens],
            "mean_decode_ms": round(1000 * sum(dec) / max(len(dec), 1), 1) if dec else None,
            "wall_s": round(time.time() - t0, 2),
            "xla_compile_delta": None,
            "kwargs": {},
        }

    result = _greedy_loop(m, ids, n_new, **kwargs)
    result["label"] = label
    result["kwargs"] = kwargs
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    model_id = sys.argv[1]
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    # Optional single-variant filter for one-process-per-variant runs:
    #   ... MODEL N VARIANTS=baseline,cpu_init_compile_cachepos,...
    want = None
    if len(sys.argv) > 3:
        want = set(sys.argv[3].split(","))

    variants = [
        ("baseline_static_greedy", {}),
        (
            "compile_old_cachepos",
            dict(use_compile=True, cpu_init_cache=False, legacy_compile=False, use_position_ids=False),
        ),
        (
            "compile_cpuinit_cachepos",
            dict(use_compile=True, cpu_init_cache=True, legacy_compile=False, use_position_ids=False),
        ),
        (
            "compile_cpuinit_posids",
            dict(use_compile=True, cpu_init_cache=True, legacy_compile=False, use_position_ids=True),
        ),
        (
            "legacy_cpuinit_cachepos",
            dict(use_compile=True, cpu_init_cache=True, legacy_compile=True, use_position_ids=False),
        ),
        (
            "legacy_cpuinit_posids",
            dict(use_compile=True, cpu_init_cache=True, legacy_compile=True, use_position_ids=True),
        ),
    ]

    # Child mode: one variant, hold the card, print RESULT JSON, exit.
    if want is not None:
        label = next(iter(want))
        kwargs = dict(next(k for n, k in variants if n == label))
        print(f"=== {label} ===", flush=True)
        row = _run_variant(label, model_id, n_new, **kwargs) if kwargs else _run_variant(label, model_id, n_new)
        print("RESULT:" + json.dumps(row), flush=True)
        return

    # Parent mode: never open the TT device — each variant is a fresh process
    # (pjrt/UMD holds CHIP_IN_USE for the process lifetime).
    import subprocess

    rows = []
    baseline_tokens = None
    for label, _kwargs in variants:
        print(f"=== {label} ===", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            model_id,
            str(n_new),
            label,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
        # 12 min hard cap per variant (compile can be slow; lock hangs are worse).
        try:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=720,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"TIMEOUT {label}", flush=True)
            rows.append({"label": label, "error": f"timeout after {exc.timeout}s"})
            continue
        if proc.stdout:
            print(proc.stdout[-3000:], flush=True)
        if proc.returncode != 0:
            err = (proc.stderr or "")[-2000:]
            print(err, flush=True)
            rows.append({"label": label, "error": err or f"exit {proc.returncode}"})
            continue
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT:"):
                payload = json.loads(line[len("RESULT:") :])
        if payload is None:
            rows.append({"label": label, "error": "no RESULT line"})
            continue
        rows.append(payload)
        if label == "baseline_static_greedy":
            baseline_tokens = payload.get("tokens")
        print(
            f"  tokens={str(payload.get('tokens'))[:60]}... "
            f"mean_decode_ms={payload.get('mean_decode_ms')} wall={payload.get('wall_s')}",
            flush=True,
        )

    # Attach parity vs baseline
    for row in rows:
        if "tokens" in row and baseline_tokens is not None and row["label"] != "baseline_static_greedy":
            bt = baseline_tokens
            rt = row["tokens"]
            match = 0
            for a, b in zip(bt, rt):
                if a != b:
                    break
                match += 1
            row["parity_vs_baseline"] = f"{match}/{len(bt)}"
            row["tokens_match"] = rt == bt

    out = {
        "model": model_id,
        "n_new": n_new,
        "baseline_tokens": baseline_tokens,
        "variants": rows,
        "notes": (
            "cpu_init = StaticCache on CPU then move buffers to XLA (tt-xla llama test pattern). "
            "legacy_compile = tt_legacy_compile=True (mutation sync path). "
            "Wrong-from-token-1 historically seen with TENSCUDA_COMPILE=1 / old cache alloc."
        ),
    }
    dest = ROOT / "results" / f"m4_compile_staticcache_{slug(model_id)}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("model", "n_new")}, indent=2))
    for row in rows:
        brief = {k: row.get(k) for k in ("label", "parity_vs_baseline", "tokens_match", "mean_decode_ms", "wall_s", "error")}
        print(brief)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
