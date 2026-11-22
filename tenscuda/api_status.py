"""Source of truth for TensCUDA API compatibility (Y / STUB / N).

Used by migration logging and ``scripts/render_api_compat.py``.
"""

from __future__ import annotations

from typing import Literal

Status = Literal["Y", "STUB", "N"]

STATUS_MEANING = {
    "Y": "Supported: remapped to TT-XLA / real device behavior",
    "STUB": "Present for script survival; semantics incomplete",
    "N": "Not supported on TensCUDA",
}

# torch.cuda.* surface patched or stubbed by tenscuda._patch
CUDA_API: dict[str, tuple[Status, str]] = {
    "is_available": ("Y", "Reports True when TT device is active"),
    "device_count": ("Y", "TT runtime device count"),
    "current_device": ("STUB", "Always 0; TT is single-card first"),
    "set_device": ("STUB", "No-op; only device 0 today"),
    "get_device_name": ("STUB", "Fixed descriptive string, not hardware query"),
    "get_device_capability": ("STUB", "Returns (8, 0) for bf16-capable scripts"),
    "get_device_properties": ("STUB", "SimpleNamespace with 32 GiB nominal total_memory"),
    "synchronize": ("Y", "maps to torch_xla.sync + xm.wait_device_ops"),
    "empty_cache": ("STUB", "No-op; TT has no CUDA-style allocator cache"),
    "is_bf16_supported": ("Y", "True on TT path"),
    "manual_seed": ("Y", "Seeds XLA RNG via xm.set_rng_state"),
    "manual_seed_all": ("Y", "Same as manual_seed on card 0"),
    "memory_allocated": ("STUB", "xm.get_memory_info when available, else 0 + STUB log"),
    "max_memory_allocated": ("STUB", "Same as memory_allocated peak tracking N/A"),
    "memory_reserved": ("STUB", "Same as memory_allocated"),
    "reset_peak_memory_stats": ("STUB", "No-op"),
    "get_rng_state": ("STUB", "Zeros tensor; avoids CUDA runtime / dynamo"),
    "set_rng_state": ("STUB", "No-op; state not round-tripped"),
    "get_rng_state_all": ("STUB", "Fake states per device"),
    "set_rng_state_all": ("STUB", "No-op"),
    "is_initialized": ("Y", "True when patch active"),
    "init": ("STUB", "No-op"),
    "current_stream": ("STUB", "Dummy stream; record/wait no-ops"),
    "default_stream": ("STUB", "Same dummy stream"),
    "stream": ("STUB", "Constructs dummy cuda.Stream equivalent"),
    "Event": ("STUB", "Dummy event; record/wait/synchronize no-op"),
}

# Tensor / Module / factory path (not under torch.cuda)
DEVICE_API: dict[str, tuple[Status, str]] = {
    "torch.device(cuda)": ("Y", "Proxy metaclass -> xla device"),
    "Tensor.cuda": ("Y", "Remap to XLA device; optional fp16->bf16"),
    "Tensor.to(cuda)": ("Y", "Device and dtype remap"),
    "Module.cuda": ("Y", "Remap parameters to XLA"),
    "Module.to(cuda)": ("Y", "Device and dtype remap"),
    "factory device=cuda kwargs": ("Y", "TorchFunctionMode in _CudaDeviceMode"),
    "torch.cuda.amp": ("N", "CUDA autocast not emulated; use bf16 weights"),
    "torch.cuda.CUDAGraph": ("N", "No CUDA graph capture"),
    "torch.cuda.graph": ("N", "No CUDA graph capture"),
    "CUDAExtension / Triton / flash-attn": ("N", "Requires CUDA kernels"),
}

ALL_ENTRIES = {f"torch.cuda.{k}": v for k, v in CUDA_API.items()} | DEVICE_API