"""Fixed-shape decode loop for Hugging Face causal LMs on TT.

HF's stock ``generate`` grows ``input_ids`` by one token per step and checks
stopping criteria on the device. Under torch-xla every new shape is a new
graph, so the stock loop triggers a fresh TT-MLIR compile for every token.

Neuron-style pattern (transformers-neuronx / NxD inference):
  * separate prefill vs single-token decode launches,
  * one XLA graph boundary per token (``_sync()`` after each forward),
  * ``StaticCache`` updated between launches, not batched inside one lazy graph.

Batching K>1 decode steps without a sync between steps corrupts ``StaticCache``
(see ``scripts/kv_diagnose.py`` and docs/PERF.md).

M4 compile unlock (overnight 2026-11-23): ``TENSCUDA_COMPILE=1`` uses
CPU-init StaticCache + torch.compile(backend="tt") with torch-function modes
exited for capture and run. Measured ~28 ms/step (~36 tok/s) on SmolLM2-135M
and ~21 ms/step (~47 tok/s) on TinyLlama with full greedy token match vs eager
baseline (``scripts/m4_compile_diverge.py``). **Default on** since Nov 2026; set
``TENSCUDA_COMPILE=0`` for the eager path. Optional E4: ``TENSCUDA_NATIVE_OPS=sdpa,rms_norm``
(opt-in; A/B 32/32, no meaningful ms win on 8B yet).
"""
from __future__ import annotations

import logging
import os
import time

import torch

log = logging.getLogger("tenscuda.generate")


def _sync(dev=None):
    import torch_xla

    torch_xla.sync()


def _xla_compile_count() -> int:
    import torch_xla.debug.metrics as met

    row = met.metric_data("CompileTime")
    return int(row[0]) if row else 0


class _ResidentDecodeIO:
    """Fixed on-device shapes for 1-token decode (M4: fewer host tensor factories)."""

    def __init__(self, dev):
        self.step_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.step_pos = torch.zeros(1, dtype=torch.long, device=dev)

    def write(self, token_id: int, cache_position: int):
        self.step_ids[0, 0] = token_id
        self.step_pos[0] = cache_position


def _compiled_forward(model):
    """torch.compile(backend="tt") once per model.

    Caller must exit TensCUDA/tt_torch torch-function modes before the first
    call so Dynamo captures a clean graph (see M4 diverge unlock).
    """
    fn = getattr(model, "_tenscuda_compiled", None)
    if fn is None:
        from ._patch import real_torch_device

        backend = os.environ.get("TENSCUDA_COMPILE_BACKEND", "tt")
        options = None
        if os.environ.get("TENSCUDA_COMPILE_LEGACY", "0") == "1":
            options = {"tt_legacy_compile": True}
        compiled = torch.compile(model, backend=backend, dynamic=False, options=options)

        def fn(*a, **k):
            with real_torch_device():
                return compiled(*a, **k)

        model._tenscuda_compiled = fn
    return fn


def _exit_trace_modes() -> list:
    """Exit TensCUDA + tt_torch torch-function modes; return contexts to re-enter."""
    import tenscuda._patch as patch

    exited: list = []
    mode = patch._S.originals.get("mode")
    if mode is not None:
        mode.__exit__(None, None, None)
        exited.append(mode)
    try:
        import tt_torch.torch_overrides as overrides

        tt = overrides.torch_function_override
        if getattr(tt, "_entered", True):
            tt.__exit__(None, None, None)
            exited.append(tt)
    except Exception:
        pass
    return exited


def _enter_trace_modes(exited: list) -> None:
    for ctx in reversed(exited):
        try:
            ctx.__enter__()
        except Exception:
            pass


def _make_static_cache(model, max_len, dev, *, cpu_init: bool):
    from transformers import StaticCache

    if not cpu_init:
        return StaticCache(config=model.config, max_cache_len=max_len)

    try:
        cache = StaticCache(
            config=model.config,
            max_cache_len=max_len,
            device="cpu",
            dtype=torch.bfloat16,
            max_batch_size=1,
        )
    except TypeError:
        try:
            cache = StaticCache(
                config=model.config, max_cache_len=max_len, device="cpu", dtype=torch.bfloat16,
            )
        except TypeError:
            cache = StaticCache(config=model.config, max_cache_len=max_len)

    if hasattr(cache, "early_initialization"):
        cfg = model.config
        n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        cache.early_initialization(
            batch_size=1, num_heads=n_heads, head_dim=head_dim,
            dtype=torch.bfloat16, device="cpu",
        )
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys.to(dev)
        if getattr(layer, "values", None) is not None:
            layer.values = layer.values.to(dev)
        cl = getattr(layer, "cumulative_length", None)
        if cl is not None and hasattr(cl, "to"):
            layer.cumulative_length = cl.to(dev)
        if hasattr(layer, "device"):
            layer.device = dev
    return cache


def _forward_prefill(fwd, model, input_ids, cache, dev):
    plen = input_ids.shape[1]
    pos = torch.arange(plen, device=dev)
    out = fwd(
        input_ids=input_ids.to(dev),
        cache_position=pos,
        past_key_values=cache,
        use_cache=True,
    )
    _sync()
    return out.logits


def _forward_decode_step(
    fwd,
    token_id: int,
    cache_position: int,
    cache,
    dev,
    *,
    decode_io: _ResidentDecodeIO | None = None,
    stats: dict | None = None,
):
    """One decode token; must end with ``_sync()`` before the next step."""
    comp_before = _xla_compile_count() if stats is not None else None
    if decode_io is not None:
        decode_io.write(token_id, cache_position)
        step_ids, step_pos = decode_io.step_ids, decode_io.step_pos
    else:
        step_ids = torch.tensor([[token_id]], device=dev)
        step_pos = torch.tensor([cache_position], device=dev)
    out = fwd(
        input_ids=step_ids,
        cache_position=step_pos,
        past_key_values=cache,
        use_cache=True,
    )
    _sync()
    if stats is not None and comp_before is not None:
        after = _xla_compile_count()
        stats.setdefault("xla_compiles_per_step", []).append(after)
        stats.setdefault("xla_compile_deltas", []).append(after - comp_before)
    return out.logits


def _compile_unlock_generate(model, ids_cpu, max_new_tokens, *, stats=None):
    """Opt-in ``TENSCUDA_COMPILE=1`` path — see ``tenscuda.compile_unlock``."""
    from tenscuda.compile_unlock import compile_decode_steps

    steps = compile_decode_steps(model, ids_cpu, max_new_tokens, use_compile=True)
    tokens = [int(s["token"]) for s in steps]
    if stats is not None:
        stats["step_seconds"] = [s["ms"] / 1000.0 for s in steps]
    dev = next(model.parameters()).device
    return torch.cat([ids_cpu, torch.tensor([tokens])], dim=1).to(dev)


def _try_vllm_delegate(model, input_ids, max_new_tokens, eos_token_id):
    from tenscuda.fast_router import decide_route, fast_mode, maybe_block_compat_generate, model_id_from_config
    from tenscuda.vllm_delegate import (
        VllmDelegateError,
        delegate_batch_greedy_generate,
        delegate_error_hint,
        delegate_greedy_generate,
        probe_vllm,
    )

    ids_cpu = input_ids.detach().to("cpu")
    decision = decide_route(
        model_id=model_id_from_config(getattr(model, "config", None)),
        batch_size=int(ids_cpu.shape[0]),
        greedy=True,
    )
    if decision.route == "delegate":
        mode = fast_mode()
        if mode in {"auto", "enforce", "aggregate"} and os.environ.get("TENSCUDA_VLLM_PROBE", "1") != "0":
            if not probe_vllm():
                if mode == "auto":
                    log.warning("tenscuda: vLLM down; falling back to TT-XLA")
                else:
                    raise RuntimeError(
                        delegate_error_hint(VllmDelegateError(f"vLLM not reachable at probe"))
                    )
                maybe_block_compat_generate(decision)
                return None
        try:
            if ids_cpu.shape[0] > 1:
                return delegate_batch_greedy_generate(
                    model,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    decision=decision,
                )
            return delegate_greedy_generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                decision=decision,
            )
        except VllmDelegateError as exc:
            if mode == "auto":
                log.warning("tenscuda: vLLM delegate failed (%s); falling back to TT-XLA", exc)
            else:
                raise RuntimeError(delegate_error_hint(exc)) from exc
    maybe_block_compat_generate(decision)
    return None


def static_greedy_generate(model, input_ids, max_new_tokens=32, eos_token_id=None,
                           attention_mask=None, pad_to=64, stats=None, compile=None):
    from tenscuda.decode_graph import get_forward_fn, pause_decode_trace_modes

    delegated = _try_vllm_delegate(model, input_ids, max_new_tokens, eos_token_id)
    if delegated is not None:
        return delegated

    ids_cpu = input_ids.detach().to("cpu")
    if ids_cpu.shape[0] != 1:
        raise NotImplementedError("tenscuda static generate: batch size 1 only")
    if attention_mask is not None and not bool((attention_mask.detach().to("cpu") == 1).all()):
        raise NotImplementedError("tenscuda static generate: padded attention_mask not supported")

    if compile is None:
        from tenscuda.env_flags import compile_enabled

        compile = compile_enabled()
    if compile:
        return _compile_unlock_generate(model, ids_cpu, max_new_tokens, stats=stats)

    dev = next(model.parameters()).device
    if v := os.environ.get("TENSCUDA_PAD_TO"):
        pad_to = int(v)
    plen = ids_cpu.shape[1]
    max_len = -(-(plen + max_new_tokens) // pad_to) * pad_to
    cpu_init = os.environ.get("TENSCUDA_STATIC_CACHE_CPU_INIT", "0") == "1"
    cache = _make_static_cache(model, max_len, dev, cpu_init=cpu_init)

    eos = set()
    if eos_token_id is not None:
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])

    fwd = get_forward_fn(model)
    decode_io = _ResidentDecodeIO(dev) if os.environ.get("TENSCUDA_RESIDENT_DECODE_IO", "0") == "1" else None
    if stats is None and os.environ.get("TENSCUDA_M4_DEBUG", "0") == "1":
        stats = {}

    out = []
    timings = []
    try:
        from monitor.activity import emit as _activity
    except Exception:
        def _activity(*_a, **_k):
            pass
    model_id = getattr(getattr(model, "config", None), "name_or_path", None) or getattr(model, "name_or_path", None)
    _activity("decoding", model=model_id, new_tokens=max_new_tokens)

    with torch.no_grad():
        t = time.time()
        logits = _forward_prefill(fwd, model, ids_cpu, cache, dev)
        nxt = int(logits[:, -1, :].argmax(-1).cpu())
        timings.append(time.time() - t)
        out.append(nxt)
        cur = plen
        while len(out) < max_new_tokens and nxt not in eos:
            t = time.time()
            with pause_decode_trace_modes():
                logits = _forward_decode_step(
                    fwd, nxt, cur, cache, dev, decode_io=decode_io, stats=stats,
                )
            nxt = int(logits[:, -1, :].argmax(-1).cpu())
            timings.append(time.time() - t)
            out.append(nxt)
            cur += 1
            dec = timings[1:]
            if dec and sum(dec) > 0 and (len(dec) == 1 or len(dec) % 2 == 0):
                _activity(
                    "decoding",
                    model=model_id,
                    tok_s=round(len(dec) / sum(dec), 2),
                    tokens=len(out),
                    step_ms=round(dec[-1] * 1000, 1),
                )

    if stats is not None:
        stats["step_seconds"] = timings
    dec = timings[1:]
    if dec and sum(dec) > 0:
        _activity("decoding", model=model_id, tok_s=round(len(dec) / sum(dec), 2), tokens=len(out))
    return torch.cat([ids_cpu, torch.tensor([out])], dim=1).to(dev)
