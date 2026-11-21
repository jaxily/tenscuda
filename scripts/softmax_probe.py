"""Probe softmax/matmul accuracy on TT vs CPU fp32 under the current compile options."""
import os, torch, torch.nn.functional as F
import tenscuda
torch.manual_seed(0)
x = torch.randn(4, 256, 512)
a = torch.randn(256, 384); m = torch.randn(384, 192)
def rel(g, r): return ((g.float() - r).norm() / r.norm()).item()
def manual(x):
    e = torch.exp(x - x.amax(-1, keepdim=True)); return e / e.sum(-1, keepdim=True)
def manual_recip(x):
    e = torch.exp(x - x.amax(-1, keepdim=True)); return e * (1.0 / e.sum(-1, keepdim=True))
ref = torch.softmax(x, -1)
print(os.environ.get("TENSCUDA_MATH_FIDELITY", "hifi4"), os.environ.get("TENSCUDA_FP32_ACC", "1"),
      "F.softmax", round(rel(torch.softmax(x.cuda(), -1).cpu(), ref), 6),
      "manual", round(rel(manual(x.cuda()).cpu(), ref), 6),
      "manual_recip", round(rel(manual_recip(x.cuda()).cpu(), ref), 6),
      "sum", round(rel(x.cuda().exp().sum(-1).cpu(), x.exp().sum(-1)), 6),
      "matmul", round(rel((a.cuda() @ m.cuda()).cpu(), a @ m), 6), flush=True)
