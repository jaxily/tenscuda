"""Offline tests for migration logging and API taxonomy (no TT card)."""
import os

import pytest

from tenscuda import api_status
from tenscuda import migration_log as mig


@pytest.fixture(autouse=True)
def _reset_migration_env():
    mig.reset_for_tests()
    old = os.environ.get("TENSCUDA_MIGRATION_LOG")
    yield
    mig.reset_for_tests()
    if old is None:
        os.environ.pop("TENSCUDA_MIGRATION_LOG", None)
    else:
        os.environ["TENSCUDA_MIGRATION_LOG"] = old


def test_api_status_covers_all_cuda_patched_names():
    expected = {
        "is_available", "device_count", "current_device", "get_device_name",
        "get_device_capability", "get_device_properties", "synchronize",
        "set_device", "empty_cache", "is_bf16_supported", "manual_seed",
        "manual_seed_all", "memory_allocated", "max_memory_allocated",
        "memory_reserved", "reset_peak_memory_stats", "is_initialized", "init",
        "get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all",
        "current_stream", "default_stream", "stream", "Event",
    }
    assert expected <= set(api_status.CUDA_API.keys())


def test_migration_log_dedupes_per_api():
    os.environ["TENSCUDA_MIGRATION_LOG"] = "2"
    mig.reset_for_tests()
    mig.record("torch.cuda.empty_cache", "stub")
    mig.record("torch.cuda.empty_cache", "stub")
    assert mig._counts["stub:torch.cuda.empty_cache"] == 2
    assert len(mig._seen) == 1


def test_migration_wrap_invokes_and_logs():
    os.environ["TENSCUDA_MIGRATION_LOG"] = "2"
    mig.reset_for_tests()
    fn = mig.wrap("torch.cuda.test", "stub", lambda: 42)
    assert fn() == 42
    assert ("torch.cuda.test", "stub") in mig._seen


def test_status_for_cuda_name_mapping():
    assert mig.status_for_cuda_name("synchronize") == "mapped"
    assert mig.status_for_cuda_name("empty_cache") == "stub"