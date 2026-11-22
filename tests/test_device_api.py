"""Run on a machine with a Tenstorrent card:  pytest -q tests/"""
import pytest
import torch

import tenscuda

pytestmark = pytest.mark.skipif(not tenscuda.is_active, reason="no Tenstorrent device")


def test_cuda_queries():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    assert "Tenstorrent" in torch.cuda.get_device_name(0)
    assert torch.cuda.is_bf16_supported()


def test_torch_device_mapping():
    d = torch.device("cuda")
    assert d.type == "xla"
    assert torch.device("cuda:0") == d
    assert isinstance(d, torch.device)
    assert torch.device("cpu").type == "cpu"  # non-CUDA devices untouched


def test_tensor_moves():
    x = torch.arange(6.0).reshape(2, 3)
    for y in (x.cuda(), x.to("cuda"), x.to(device="cuda"), x.to(torch.device("cuda"))):
        assert y.device.type == "xla"
        assert torch.equal(y.cpu(), x)


def test_factories():
    z = torch.zeros(4, device="cuda")
    assert z.device.type == "xla"
    assert torch.ones(3, device=torch.device("cuda")).cpu().sum().item() == 3


def test_module_cuda_and_fp16_maps_to_bf16():
    m = torch.nn.Linear(8, 4).half().cuda()
    p = next(m.parameters())
    assert p.device.type == "xla" and p.dtype is torch.bfloat16


def test_matmul_close_to_cpu():
    torch.manual_seed(0)
    a, b = torch.randn(64, 64), torch.randn(64, 64)
    got = (a.cuda() @ b.cuda()).cpu()
    # TT computes this in bf16-class precision
    assert torch.allclose(got, a @ b, atol=0.25, rtol=0.05)


def test_synchronize():
    torch.cuda.synchronize()


def test_stream_stubs():
    s = torch.cuda.current_stream()
    s.synchronize()
    e = torch.cuda.Event()
    e.record()
    e.synchronize()


def test_migration_log_on_stub_touch():
    import os

    from tenscuda import migration_log as mig

    mig.reset_for_tests()
    os.environ["TENSCUDA_MIGRATION_LOG"] = "2"
    torch.cuda.memory_allocated()
    assert any(k.endswith("memory_allocated") for k in mig._counts)
