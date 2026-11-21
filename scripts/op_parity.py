"""Op-level parity: TT (via tenscuda) vs CPU. Writes results/op_parity.json.

Error metrics per op and dtype:
  max_abs  = max |tt - cpu|
  max_rel  = max |tt - cpu| / (|cpu| + 1e-6)  over elements with |cpu| > 1e-2
  rel_l2   = ||tt - cpu|| / ||cpu||
CPU reference is computed in float32 from the same (dtype-rounded) inputs.
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

import tenscuda  # noqa: F401
import torch_xla.debug.metrics as met

torch.manual_seed(0)


def rms_norm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def cases(dt):
    B, S, H, D = 1, 64, 8, 64
    x = torch.randn(4, 256, 512).to(dt)
    w = torch.randn(1024, 512).to(dt) / 512 ** 0.5
    b = torch.randn(1024).to(dt)
    g = (1 + 0.1 * torch.randn(512)).to(dt)
    q, k, v = (torch.randn(B, H, S, D).to(dt) for _ in range(3))
    a = torch.randn(256, 384).to(dt); m = torch.randn(384, 192).to(dt)
    ids = torch.randint(0, 1000, (2, 32)); emb = torch.randn(1000, 256).to(dt)
    return {
        "matmul_256x384x192": (lambda a, m: a @ m, (a, m)),
        "linear_4x256x512->1024": (lambda x, w, b: F.linear(x, w, b), (x, w, b)),
        "bmm_attn_scores": (lambda q, k: q @ k.transpose(-1, -2), (q, k)),
        "layer_norm_512": (lambda x, g: F.layer_norm(x, (512,), g, None, 1e-5), (x, g)),
        "rms_norm_manual_512": (rms_norm, (x, g)),
        "rms_norm_F_512": (lambda x, g: F.rms_norm(x, (512,), g, 1e-6), (x, g)),
        "softmax_lastdim": (lambda x: torch.softmax(x, -1), (x,)),
        "sdpa_causal_8h_64": (lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True), (q, k, v)),
        "silu": (F.silu, (x,)),
        "gelu_tanh": (lambda x: F.gelu(x, approximate="tanh"), (x,)),
        "exp": (torch.exp, (x.clamp(-10, 10),)),
        "embedding_gather": (lambda i, e: F.embedding(i, e), (ids, emb)),
        "cumsum": (lambda x: torch.cumsum(x, -1), (x,)),
        "argmax": (lambda x: torch.argmax(x, -1).float(), (x,)),
    }


def err(got, ref):
    got, ref = got.float(), ref.float()
    d = (got - ref).abs()
    mask = ref.abs() > 1e-2
    rel = (d[mask] / ref.abs()[mask]).max().item() if mask.any() else 0.0
    return {"max_abs": d.max().item(), "max_rel": rel,
            "rel_l2": (d.norm() / ref.norm().clamp_min(1e-12)).item()}


out = {"compile_options": getattr(tenscuda._patch._S, "compile_options", {}), "ops": []}
dtypes = [torch.float32, torch.bfloat16] if len(sys.argv) < 2 else [getattr(torch, d) for d in sys.argv[1:]]
for dt in dtypes:
    for name, (fn, args) in cases(dt).items():
        ref = fn(*[t.float() if t.is_floating_point() else t for t in args])
        t0 = time.time()
        try:
            got = fn(*[t.cuda() for t in args]).cpu()
            r = {"op": name, "dtype": str(dt).split(".")[-1], "seconds_incl_compile": round(time.time() - t0, 2),
                 **err(got, ref)}
        except Exception as e:  # record and keep going
            r = {"op": name, "dtype": str(dt).split(".")[-1], "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(json.dumps(r), flush=True)
        out["ops"].append(r)

# Host <-> device transfer must be bit-exact.
for dt in (torch.float32, torch.bfloat16, torch.int32, torch.int64):
    x = (torch.randn(1024, 1024) * 100).to(dt)
    t0 = time.time(); y = x.cuda(); torch.cuda.synchronize(); h2d = time.time() - t0
    t0 = time.time(); z = y.cpu(); d2h = time.time() - t0
    r = {"op": "h2d_d2h_roundtrip_1Mi", "dtype": str(dt).split(".")[-1], "bit_exact": bool(torch.equal(x, z)),
         "h2d_s": round(h2d, 4), "d2h_s": round(d2h, 4)}
    print(json.dumps(r), flush=True); out["ops"].append(r)

out["cpu_fallback_ops"] = sorted(met.executed_fallback_ops())
out["xla_compiles"] = (met.metric_data("CompileTime") or [0])[0]
print("cpu_fallback_ops:", out["cpu_fallback_ops"])
json.dump(out, open("results/op_parity.json", "w"), indent=1)
