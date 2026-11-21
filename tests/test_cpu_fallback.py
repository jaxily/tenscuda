"""Ops torch-xla has no lowering for run on the host CPU instead of failing."""
import pytest
import torch

import tenscuda

pytestmark = pytest.mark.skipif(not tenscuda.is_active, reason="no Tenstorrent device")


def test_unlowered_op_falls_back_to_cpu():
    import torch_xla.debug.metrics as met

    x = torch.tensor([3, 1, 3, 2, 1]).cuda()
    u = torch.unique(x)            # data-dependent output shape
    assert u.cpu().tolist() == [1, 2, 3]
    nz = torch.nonzero(x > 1)
    assert nz.cpu().flatten().tolist() == [0, 2, 3]
    print("fallback ops:", met.executed_fallback_ops())
