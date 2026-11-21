"""Block-float (bfp8 / bfp4) weights via TT-XLA's weight-dtype pass.

Measured with scripts/quant_probe.py: in lazy mode ``torch.ops.tt.weight_dtype_override``
does nothing by itself. tt-mlir only converts weights whose graph argument is marked as
``parameter``. Each forward therefore applies override, then mark_argument_attributes(parameter).

Caveat: torch-xla's in-process graph cache doesn't key on the dtype attribute, so do not
switch weight dtypes for the same model in one process. A structurally identical graph
reuses the first executable.
"""
from __future__ import annotations

import fnmatch

import torch
from torch.nn.utils import parametrize


class _BfpWeight(torch.nn.Module):
    def __init__(self, dtype_str: str, name: str):
        super().__init__()
        self.dtype_str, self.name = dtype_str, name

    def forward(self, w):
        w = torch.ops.tt.weight_dtype_override(w, self.dtype_str)
        return torch.ops.tt.mark_argument_attributes(w, "parameter", self.name)


def quantize_weights(model, dtype_str="bfp_bf8", include=("model.layers.*",), exclude=()):
    """Mark matching nn.Linear weights for bfp conversion. Returns the list of weight names."""
    import tt_torch  # noqa: F401  (registers torch.ops.tt.*)

    assert dtype_str in ("bfp_bf8", "bfp_bf4", "bf16")
    done = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not any(fnmatch.fnmatch(name, p) for p in include) or any(fnmatch.fnmatch(name, p) for p in exclude):
            continue
        parametrize.register_parametrization(mod, "weight", _BfpWeight(dtype_str, f"{name}.weight"))
        done.append(f"{name}.weight")
    return done
