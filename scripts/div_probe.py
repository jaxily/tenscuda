"""Which division forms are inaccurate on TT?"""
import torch, torch.nn.functional as F
import tenscuda
torch.manual_seed(0)
def rel(g, r): return ((g.float() - r).norm() / r.norm()).item()
x = torch.rand(4, 256, 512) + 0.1; y = torch.rand(4, 256, 512) + 0.5; s = torch.rand(4, 256, 1) * 100 + 1
X, Y, S = x.cuda(), y.cuda(), s.cuda()
r = {}
r["div_same"] = rel((X / Y).cpu(), x / y)
r["div_bcast"] = rel((X / S).cpu(), x / s)
r["recip"] = rel(torch.reciprocal(S).cpu(), 1 / s)
r["mul_recip_bcast"] = rel((X * torch.reciprocal(S)).cpu(), x / s)
r["div_scalar"] = rel((X / 7.0).cpu(), x / 7.0)
r["rsqrt"] = rel(torch.rsqrt(Y).cpu(), torch.rsqrt(y))
q = torch.randn(1, 8, 64, 64); k = torch.randn(1, 8, 64, 64); v = torch.randn(1, 8, 64, 64)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
r["sdpa"] = rel(F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True).cpu(), ref)
for k_, v_ in r.items(): print(f"{k_:18s} {v_:.6f}", flush=True)
