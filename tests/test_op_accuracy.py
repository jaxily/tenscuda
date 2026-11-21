"""Op accuracy vs CPU fp32 (thresholds from results/op_parity.json, with headroom)."""
import pytest
import torch
import torch.nn.functional as F

import tenscuda

pytestmark = pytest.mark.skipif(not tenscuda.is_active, reason="no Tenstorrent device")


def rel(g, r):
    return ((g.float().cpu() - r).norm() / r.norm()).item()


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_softmax_accurate():
    x = torch.randn(4, 256, 512)
    assert rel(torch.softmax(x.cuda(), -1), torch.softmax(x, -1)) < 2e-3   # fused ttnn softmax: 2.3e-2


def test_tensor_softmax_method():
    x = torch.randn(8, 64)
    assert rel(x.cuda().softmax(-1), x.softmax(-1)) < 2e-3


def test_sdpa_causal_accurate():
    q, k, v = (torch.randn(1, 8, 64, 64) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
    assert rel(got, ref) < 1e-2                                               # fused: 2.4e-2


def test_sdpa_gqa_and_mask():
    q = torch.randn(1, 8, 16, 32); k = torch.randn(1, 2, 16, 32); v = torch.randn(1, 2, 16, 32)
    mask = torch.ones(16, 16, dtype=torch.bool).tril()
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    got = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), attn_mask=mask.cuda(), enable_gqa=True)
    assert rel(got, ref) < 1e-2


def test_matmul_fp32_tf32_class():
    a, b = torch.randn(256, 384), torch.randn(384, 192)
    assert rel(a.cuda() @ b.cuda(), a @ b) < 3e-3   # ~1.2e-3 measured (HiFi4, fp32 acc)


def test_rms_norm():
    x = torch.randn(4, 64, 512); w = torch.ones(512)
    assert rel(F.rms_norm(x.cuda(), (512,), w.cuda(), 1e-6), F.rms_norm(x, (512,), w, 1e-6)) < 5e-3


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16, torch.int32, torch.int64])
def test_transfer_roundtrip_bit_exact(dt):
    x = (torch.randn(512, 512) * 100).to(dt)
    assert torch.equal(x.cuda().cpu(), x)
