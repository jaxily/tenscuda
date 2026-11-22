"""M4 B3: decode graph helpers — pause trace modes, thin forward binding.

- ``TENSCUDA_DECODE_PAUSE_MODES=1`` — exit torch-function modes during decode only.
- ``TENSCUDA_THIN_DECODE=1`` — call cached ``model.forward`` instead of ``model(...)``.

Parity must pass ``scripts/kv_diagnose.py`` / ``scripts/m4_pause_parity_gate.py``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable
from typing import Iterator


def get_forward_fn(model: Any) -> Callable[..., Any]:
    """B3 thin wrapper: reuse bound ``forward`` (avoids extra ``Module.__call__`` hooks)."""
    if os.environ.get("TENSCUDA_THIN_DECODE", "0") != "1":
        return model
    fn = getattr(model, "_tenscuda_forward_fn", None)
    if fn is None:
        fn = model.forward
        model._tenscuda_forward_fn = fn
    return fn


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