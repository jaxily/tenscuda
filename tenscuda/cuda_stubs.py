"""Survivability stubs for torch.cuda stream/event surface."""
from __future__ import annotations

from .migration_log import record, wrap


class _CudaStreamStub:
    def __init__(self, *args, **kwargs):
        record("torch.cuda.stream", "stub", "Dummy stream")

    def wait_event(self, *args, **kwargs):
        return None

    def wait_stream(self, *args, **kwargs):
        return None

    def record_event(self, *args, **kwargs):
        return None

    def synchronize(self):
        from ._patch import sync

        sync()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _CudaEventStub:
    def __init__(self, *args, **kwargs):
        record("torch.cuda.Event", "stub", "Dummy event")

    def record(self, *args, **kwargs):
        return None

    def wait(self, *args, **kwargs):
        return None

    def synchronize(self):
        from ._patch import sync

        sync()

    def query(self):
        return True

    def elapsed_time(self, *args, **kwargs):
        return 0.0


def current_stream(device=None):
    return _CudaStreamStub()


def default_stream(device=None):
    return _CudaStreamStub()


def stream(stream_obj):
    return stream_obj


def memory_bytes_stub(which: str, device=None) -> int:
    """Try xm.get_memory_info; else 0 with STUB log."""
    try:
        import torch_xla.core.xla_model as xm

        info = xm.get_memory_info(device)
        if info is None:
            raise RuntimeError("no memory info")
        if which == "allocated":
            return int(info.get("bytes_used", info.get("bytesUsed", 0)))
        if which == "reserved":
            return int(info.get("bytes_limit", info.get("bytesLimit", 0)))
        if which == "max_allocated":
            return int(info.get("bytes_used", info.get("bytesUsed", 0)))
    except Exception:
        record(f"torch.cuda.memory_{which}", "stub", "pjrt get_memory_info unavailable")
    return 0


def patch_stream_api(cuda_module) -> None:
    cuda_module.Stream = _CudaStreamStub
    cuda_module.Event = _CudaEventStub
    cuda_module.current_stream = wrap("torch.cuda.current_stream", "stub", current_stream)
    cuda_module.default_stream = wrap("torch.cuda.default_stream", "stub", default_stream)
    cuda_module.stream = wrap("torch.cuda.stream", "stub", stream)