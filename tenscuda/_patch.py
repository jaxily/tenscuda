"""Monkey-patches that map torch's CUDA device API onto the TT XLA device."""
from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from . import accurate_ops as _accurate
from . import native_ops as _native
from . import cuda_stubs
from . import migration_log as mig

log = logging.getLogger("tenscuda")

_CUDA_RE = re.compile(r"^cuda(?::(\d+))?$")


@dataclass
class _State:
    active: bool = False
    device: object = None
    n_devices: int = 0
    fp16_to_bf16: bool = True
    exact_softmax: bool = True
    compile_options: dict = field(default_factory=dict)
    originals: dict = field(default_factory=dict)


_S = _State()


# ----------------------------------------------------------------- helpers
def _xla_device(index: int = 0):
    import torch_xla

    return torch_xla.device(index)


def _map_device(dev):
    """Translate a CUDA device spec into the matching XLA device; else unchanged."""
    if dev is None:
        return None
    if isinstance(dev, int):  # torch treats bare ints as CUDA ordinals
        return _xla_device(dev)
    if isinstance(dev, str):
        m = _CUDA_RE.match(dev.strip())
        if m:
            return _xla_device(int(m.group(1) or 0))
        return dev
    real_device = _S.originals.get("torch.device", torch.device)
    if isinstance(dev, real_device) and dev.type == "cuda":
        return _xla_device(dev.index or 0)
    return dev


def _map_dtype(dt):
    if _S.fp16_to_bf16 and dt is torch.float16:
        return torch.bfloat16
    return dt


def _fix_kwargs(kw):
    if "device" in kw:
        kw["device"] = _map_device(kw["device"])
    if "dtype" in kw and _S.fp16_to_bf16:
        if isinstance(kw.get("device"), torch.device) and kw["device"].type == "xla":
            kw["dtype"] = _map_dtype(kw["dtype"])
    return kw


def _is_xla(dev) -> bool:
    return isinstance(dev, torch.device) and dev.type == "xla"


class real_torch_device:
    """Temporarily restore the real ``torch.device`` class.

    torch.fx codegen compares ``obj != torch.device`` to decide whether to bind
    the ``device`` name in generated code; with our proxy installed it emits
    ``device(type='xla')`` without binding it (NameError). Used around
    torch.compile calls.
    """

    def __enter__(self):
        self._saved = torch.device
        torch.device = _S.originals.get("torch.device", torch.device)

    def __exit__(self, *exc):
        torch.device = self._saved


def sync():
    """Equivalent of torch.cuda.synchronize(): flush and wait for the device."""
    import torch_xla
    import torch_xla.core.xla_model as xm

    torch_xla.sync()
    xm.wait_device_ops()


# --------------------------------------------------------- torch.device shim
class _DeviceMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _S.originals["torch.device"])

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _S.originals["torch.device"])

    def __call__(cls, *args, **kw):
        real = _S.originals["torch.device"]
        if args and isinstance(args[0], str) and _CUDA_RE.match(args[0].strip()):
            idx = args[1] if len(args) > 1 else kw.get("index")
            m = _CUDA_RE.match(args[0].strip())
            idx = idx if idx is not None else int(m.group(1) or 0)
            return _xla_device(idx)
        return real(*args, **kw)


def _make_device_proxy():
    class device(metaclass=_DeviceMeta):  # noqa: N801  (mirrors torch.device)
        """TensCUDA proxy: torch.device('cuda') -> TT XLA device."""

    return device


# ------------------------------------------------------- tensor/module moves
def _patch_tensor_and_module():
    T, M = torch.Tensor, torch.nn.Module
    o = _S.originals
    o["Tensor.to"], o["Tensor.cuda"] = T.to, T.cuda
    o["Module.to"], o["Module.cuda"] = M.to, M.cuda

    def _map_to_args(args, kw):
        args = list(args)
        if args and not isinstance(args[0], torch.Tensor):
            args[0] = _map_device(args[0]) if not isinstance(args[0], torch.dtype) else args[0]
        kw = _fix_kwargs(dict(kw))
        target = args[0] if args else kw.get("device")
        if _S.fp16_to_bf16 and _is_xla(target):
            args = [_map_dtype(a) if isinstance(a, torch.dtype) else a for a in args]
            if "dtype" in kw:
                kw["dtype"] = _map_dtype(kw["dtype"])
        return args, kw

    @functools.wraps(o["Tensor.to"])
    def tensor_to(self, *args, **kw):
        args, kw = _map_to_args(args, kw)
        return o["Tensor.to"](self, *args, **kw)

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        dev = _xla_device(0) if device is None else _map_device(device)
        out = o["Tensor.to"](self, dev)
        if _S.fp16_to_bf16 and out.dtype is torch.float16:
            out = o["Tensor.to"](out, torch.bfloat16)
        return out

    @functools.wraps(o["Module.to"])
    def module_to(self, *args, **kw):
        args, kw = _map_to_args(args, kw)
        target = args[0] if args else kw.get("device")
        out = o["Module.to"](self, *args, **kw)
        if _is_xla(target) and _S.fp16_to_bf16:
            _fp16_params_to_bf16(out)
        return out

    def module_cuda(self, device=None):
        dev = _xla_device(0) if device is None else _map_device(device)
        out = o["Module.to"](self, dev)
        if _S.fp16_to_bf16:
            _fp16_params_to_bf16(out)
        return out

    T.to, T.cuda = tensor_to, tensor_cuda
    M.to, M.cuda = module_to, module_cuda


def _fp16_params_to_bf16(module):
    if any(p.dtype is torch.float16 for p in module.parameters()):
        _S.originals["Module.to"](module, torch.bfloat16)


def _is_cuda_spec(dev) -> bool:
    if isinstance(dev, str):
        return bool(_CUDA_RE.match(dev.strip()))
    real_device = _S.originals.get("torch.device", torch.device)
    return isinstance(dev, real_device) and dev.type == "cuda"


class _CudaDeviceMode(torch.overrides.TorchFunctionMode):
    """Rewrites ``device="cuda"`` kwargs on *every* torch call (factories included).

    A TorchFunctionMode is used instead of wrapping ``torch.zeros`` & co. in
    Python: wrapping them breaks torch's own ``with torch.device(...)`` context
    (it keys on the original builtins), which silently left some HF weights
    uninitialised in an early version and changed CPU results.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if os.environ.get("TENSCUDA_NATIVE_OPS"):
            handled, res = _native.maybe_override(func, args, kwargs)
            if handled:
                return res
        skip_sdpa = os.environ.get("TENSCUDA_NATIVE_OPS", "").strip().lower() in (
            "1", "all", "true", "yes", "sdpa",
        ) or "sdpa" in {p.strip() for p in os.environ.get("TENSCUDA_NATIVE_OPS", "").split(",")}
        if _S.exact_softmax and (
            func in _accurate.SOFTMAX_FUNCS
            or (func in _accurate.SDPA_FUNCS and not skip_sdpa)
        ):
            handled, res = _accurate.maybe_override(func, args, kwargs)
            if handled:
                return res
        if kwargs and "device" in kwargs and _is_cuda_spec(kwargs["device"]):
            kwargs = dict(kwargs)
            kwargs["device"] = _map_device(kwargs["device"])
            if _S.fp16_to_bf16 and kwargs.get("dtype") is torch.float16:
                kwargs["dtype"] = torch.bfloat16
        return func(*args, **(kwargs or {}))


def _patch_factories():
    mode = _CudaDeviceMode()
    mode.__enter__()
    _S.originals["mode"] = mode


def _seed_xla(seed):
    try:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(int(seed))
    except Exception:
        pass


# ---------------------------------------------------------------- torch.cuda
def _patch_torch_cuda():
    c = torch.cuda
    names = ("is_available", "device_count", "current_device", "get_device_name",
             "get_device_capability", "get_device_properties", "synchronize",
             "set_device", "empty_cache", "is_bf16_supported", "manual_seed",
             "manual_seed_all", "memory_allocated", "max_memory_allocated",
             "memory_reserved", "reset_peak_memory_stats", "is_initialized", "init",
             "get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all",
             "current_stream", "default_stream", "stream", "Stream", "Event")
    for n in names:
        if hasattr(c, n):
            _S.originals[f"cuda.{n}"] = getattr(c, n)

    n = _S.n_devices
    W = mig.wrap

    def is_available():
        return True

    def device_count():
        return n

    def current_device():
        return 0

    def set_device(d):
        return None

    def get_device_name(d=None):
        return "Tenstorrent Blackhole (TensCUDA/TT-XLA)"

    def get_device_capability(d=None):
        return (8, 0)

    def get_device_properties(d=None):
        return SimpleNamespace(
            name="Tenstorrent Blackhole (TensCUDA/TT-XLA)", total_memory=32 * 1024**3,
            major=8, minor=0, multi_processor_count=120)

    def empty_cache():
        return None

    def is_bf16_supported(*a, **k):
        return True

    def reset_peak_memory_stats(d=None):
        return None

    def get_rng_state(device="cuda"):
        return torch.zeros(16, dtype=torch.uint8)

    def set_rng_state(state, device="cuda"):
        return None

    def get_rng_state_all():
        return [torch.zeros(16, dtype=torch.uint8)] * n

    def set_rng_state_all(states):
        return None

    def is_initialized():
        return True

    def init():
        return None

    c.is_available = W("torch.cuda.is_available", "mapped", is_available)
    c.device_count = W("torch.cuda.device_count", "mapped", device_count)
    c.current_device = W("torch.cuda.current_device", "stub", current_device)
    c.set_device = W("torch.cuda.set_device", "stub", set_device)
    c.get_device_name = W("torch.cuda.get_device_name", "stub", get_device_name)
    c.get_device_capability = W("torch.cuda.get_device_capability", "stub", get_device_capability)
    c.get_device_properties = W("torch.cuda.get_device_properties", "stub", get_device_properties)
    c.synchronize = W("torch.cuda.synchronize", "mapped", lambda d=None: sync())
    c.empty_cache = W("torch.cuda.empty_cache", "stub", empty_cache)
    c.is_bf16_supported = W("torch.cuda.is_bf16_supported", "mapped", is_bf16_supported)
    c.manual_seed = W("torch.cuda.manual_seed", "mapped", _seed_xla)
    c.manual_seed_all = W("torch.cuda.manual_seed_all", "mapped", _seed_xla)
    c.memory_allocated = W(
        "torch.cuda.memory_allocated", "stub",
        lambda d=None: cuda_stubs.memory_bytes_stub("allocated", d),
    )
    c.max_memory_allocated = W(
        "torch.cuda.max_memory_allocated", "stub",
        lambda d=None: cuda_stubs.memory_bytes_stub("max_allocated", d),
    )
    c.memory_reserved = W(
        "torch.cuda.memory_reserved", "stub",
        lambda d=None: cuda_stubs.memory_bytes_stub("reserved", d),
    )
    c.reset_peak_memory_stats = W("torch.cuda.reset_peak_memory_stats", "stub", reset_peak_memory_stats)
    c.get_rng_state = W("torch.cuda.get_rng_state", "stub", get_rng_state)
    c.set_rng_state = W("torch.cuda.set_rng_state", "stub", set_rng_state)
    c.get_rng_state_all = W("torch.cuda.get_rng_state_all", "stub", get_rng_state_all)
    c.set_rng_state_all = W("torch.cuda.set_rng_state_all", "stub", set_rng_state_all)
    c.is_initialized = W("torch.cuda.is_initialized", "mapped", is_initialized)
    c.init = W("torch.cuda.init", "stub", init)
    cuda_stubs.patch_stream_api(c)


# --------------------------------------------------- HF generate: static cache
_STATIC_OK_KW = {"max_new_tokens", "do_sample", "num_beams", "pad_token_id", "eos_token_id",
                 "attention_mask", "generation_config", "use_cache", "temperature", "top_p",
                 "top_k"}


def _patch_hf_generate():
    """Decode with a fixed-shape KV cache on TT so each step reuses one compiled graph.

    With HF's default dynamic cache every new token changes tensor shapes and
    forces a fresh TT compile per token.
    """
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:  # transformers not installed: nothing to do
        return
    orig = GenerationMixin.generate
    _S.originals["hf.generate"] = orig

    @functools.wraps(orig)
    def generate(self, input_ids=None, *args, **kw):
        try:
            on_tt = _is_xla(next(self.parameters()).device)
        except StopIteration:
            on_tt = False
        if not on_tt or args or os.environ.get("TENSCUDA_HF_GENERATE") == "1":
            return orig(self, input_ids, *args, **kw)
        gc = kw.get("generation_config") or self.generation_config
        greedy = not kw.get("do_sample", getattr(gc, "do_sample", False)) and \
            kw.get("num_beams", getattr(gc, "num_beams", 1) or 1) == 1
        unsupported = set(kw) - _STATIC_OK_KW
        if greedy and input_ids is not None and input_ids.shape[0] == 1 and not unsupported:
            from .fast_router import decide_route, maybe_block_compat_generate, model_id_from_config

            mid = model_id_from_config(getattr(self, "config", None))
            decision = decide_route(
                model_id=mid,
                batch_size=int(input_ids.shape[0]),
                greedy=True,
            )
            maybe_block_compat_generate(decision)

            from .generate import static_greedy_generate

            mnt = kw.get("max_new_tokens") or getattr(gc, "max_new_tokens", None) or 20
            eos = kw.get("eos_token_id", getattr(gc, "eos_token_id", None))
            return static_greedy_generate(self, input_ids, max_new_tokens=mnt, eos_token_id=eos,
                                          attention_mask=kw.get("attention_mask"))
        log.warning("tenscuda: falling back to stock HF generate (slow on TT: "
                    "a new compile per token). unsupported=%s greedy=%s", sorted(unsupported), greedy)
        if "past_key_values" not in kw and "cache_implementation" not in kw \
                and getattr(gc, "cache_implementation", None) is None:
            kw["cache_implementation"] = "static"
        return orig(self, input_ids, *args, **kw)

    GenerationMixin.generate = generate


def _set_compile_options():
    """Favour CUDA-like numerics: HiFi4 math and fp32 accumulation by default.

    Override with TENSCUDA_MATH_FIDELITY=lofi|hifi2|hifi3|hifi4 and
    TENSCUDA_FP32_ACC=0|1. Extra raw options: TENSCUDA_COMPILE_OPTIONS="k=v,k=v".
    """
    import torch_xla

    opts = {
        "math_fidelity": os.environ.get("TENSCUDA_MATH_FIDELITY", "hifi4"),
        "fp32_dest_acc_en": "true" if os.environ.get("TENSCUDA_FP32_ACC", "1") != "0" else "false",
    }
    for kv in filter(None, os.environ.get("TENSCUDA_COMPILE_OPTIONS", "").split(",")):
        k, _, v = kv.partition("=")
        opts[k.strip()] = v.strip()
    try:
        torch_xla.set_custom_compile_options(opts)
        _S.compile_options = opts
    except Exception as e:  # older torch-xla
        log.warning("tenscuda: could not set TT compile options %s: %s", opts, e)


# ------------------------------------------------------------- install/undo
def install(fp16_to_bf16: bool = True, static_cache: bool = True) -> _State:
    if _S.active:
        return _S
    try:
        import torch_xla  # noqa: F401
        import torch_xla.runtime as xr

        n = xr.global_runtime_device_count()
        dtype = xr.device_type()
    except Exception as e:  # no plugin / no card: stay a no-op
        log.warning("tenscuda: no Tenstorrent XLA device (%s); leaving torch untouched", e)
        return _S
    if not n or dtype not in ("TT",):
        log.warning("tenscuda: XLA device type %r is not TT; leaving torch untouched", dtype)
        return _S

    _S.n_devices, _S.fp16_to_bf16 = n, fp16_to_bf16
    _S.exact_softmax = os.environ.get("TENSCUDA_EXACT_SOFTMAX", "1") != "0"
    _set_compile_options()
    _S.originals["torch.device"] = torch.device
    _patch_torch_cuda()
    _patch_tensor_and_module()
    _patch_factories()
    torch.device = _make_device_proxy()
    if static_cache:
        _patch_hf_generate()
    _S.device = _xla_device(0)
    _S.active = True
    mig.register_summary_on_exit()
    if os.environ.get("TENSCUDA_QUIET", "0") != "1":
        log.warning("tenscuda: 'cuda' -> %s (%d TT device(s))", _S.device, n)
    return _S


def uninstall():
    """Restore every patched attribute (mainly for tests)."""
    o = _S.originals
    if not _S.active:
        return
    for k, v in o.items():
        if k == "torch.device":
            torch.device = v
        elif k == "mode":
            v.__exit__(None, None, None)
        elif k.startswith("cuda."):
            setattr(torch.cuda, k[5:], v)
        elif k == "Tensor.to":
            torch.Tensor.to = v
        elif k == "Tensor.cuda":
            torch.Tensor.cuda = v
        elif k == "Module.to":
            torch.nn.Module.to = v
        elif k == "Module.cuda":
            torch.nn.Module.cuda = v
        elif k == "hf.generate":
            from transformers.generation.utils import GenerationMixin

            GenerationMixin.generate = v
    o.clear()
    _S.active = False
