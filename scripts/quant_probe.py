"""Is bfp8/bfp4 weight conversion actually applied? Single-Linear probe.

For y = x @ W^T with W 2048x2048, compare TT output against CPU fp32 for each
annotation variant. A real bfp4 conversion must show clearly larger error than
bfp8, and bfp8 larger than bf16. Also prints device memory info if exposed.
"""
import sys, torch
import tenscuda
import torch_xla, torch_xla.core.xla_model as xm
from tt_torch.weight_dtype import WeightDtypeParametrization  # noqa: F401
torch.manual_seed(0)
x = torch.randn(32, 2048); W = torch.randn(2048, 2048) / 45
ref = x @ W.T
def rel(g): return ((g.float().cpu() - ref).norm() / ref.norm()).item()
X = x.to(torch.bfloat16).cuda(); Wd = W.to(torch.bfloat16).cuda()
variants = {
  "bf16": lambda w: w,
  "override_bf8": lambda w: torch.ops.tt.weight_dtype_override(w, "bfp_bf8"),
  "override_bf4": lambda w: torch.ops.tt.weight_dtype_override(w, "bfp_bf4"),
  "param+override_bf8": lambda w: torch.ops.tt.weight_dtype_override(torch.ops.tt.mark_argument_attributes(w, "parameter", "w"), "bfp_bf8"),
  "param+override_bf4": lambda w: torch.ops.tt.weight_dtype_override(torch.ops.tt.mark_argument_attributes(w, "parameter", "w"), "bfp_bf4"),
  "override_bf4+param": lambda w: torch.ops.tt.mark_argument_attributes(torch.ops.tt.weight_dtype_override(w, "bfp_bf4"), "parameter", "w"),
  "override_bf8+param": lambda w: torch.ops.tt.mark_argument_attributes(torch.ops.tt.weight_dtype_override(w, "bfp_bf8"), "parameter", "w"),
  "override_bf16+param": lambda w: torch.ops.tt.mark_argument_attributes(torch.ops.tt.weight_dtype_override(w, "bf16"), "parameter", "w"),
  "param_only": lambda w: torch.ops.tt.mark_argument_attributes(w, "parameter", "w"),
  "const_only": lambda w: torch.ops.tt.mark_argument_attributes(w, "constant", "w"),
}
which = sys.argv[1:] or list(variants)
for k in which:
    y = torch.nn.functional.linear(X, variants[k](Wd))
    print(f"QPROBE {k:22s} rel_err={rel(y):.5f}", flush=True)
try:
    print("MEMINFO", xm.get_memory_info(torch_xla.device()))
except Exception as e:
    print("MEMINFO unavailable:", type(e).__name__, str(e)[:150])
