"""TensCUDA: run unmodified CUDA-style PyTorch code on Tenstorrent Blackhole.

``import tenscuda`` before your script's own torch code. Calls such as
``torch.cuda.is_available()``, ``.cuda()``, ``.to("cuda")`` and
``torch.device("cuda")`` are redirected to the Tenstorrent device exposed by
TT-XLA (torch-xla + the ``pjrt-plugin-tt`` PJRT plugin).

TensCUDA does not translate CUDA kernels. It maps PyTorch's CUDA *device API*
onto an XLA device, and TT-XLA compiles the resulting graphs for the card.
Ops that torch-xla cannot lower run on the host CPU (torch-xla's built-in
aten fallback).
"""
from __future__ import annotations

import os

__version__ = "0.0.1"

from . import _patch  # noqa: E402

_state = _patch.install(
    fp16_to_bf16=os.environ.get("TENSCUDA_FP16_TO_BF16", "1") != "0",
    static_cache=os.environ.get("TENSCUDA_STATIC_CACHE", "1") != "0",
)

device = _state.device          # the real torch_xla device
is_active = _state.active       # False if no TT device was found (pure no-op)
uninstall = _patch.uninstall
sync = _patch.sync
