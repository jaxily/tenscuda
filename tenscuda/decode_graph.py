"""M4 B3a: optional pause of torch-function trace modes during decode only.

Set ``TENSCUDA_DECODE_PAUSE_MODES=1`` to exit ``_CudaDeviceMode`` and
``tt_torch`` overrides for the single-token decode loop (prefill keeps modes on).
Parity must pass ``scripts/kv_diagnose.py`` before enabling by default.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def pause_decode_trace_modes() -> Iterator[None]:
    if os.environ.get("TENSCUDA_DECODE_PAUSE_MODES") != "1":
        yield
        return

    import tenscuda._patch as patch

    exited: list[tuple[str, object]] = []
    mode = patch._S.originals.get("mode")
    if mode is not None:
        mode.__exit__(None, None, None)
        exited.append(("tenscuda_mode", mode))
    try:
        import tt_torch.torch_overrides as overrides

        tt = overrides.torch_function_override
        if getattr(tt, "_entered", True):
            tt.__exit__(None, None, None)
            exited.append(("tt_torch", tt))
    except Exception:
        pass
    try:
        yield
    finally:
        for _name, ctx in reversed(exited):
            ctx.__enter__()