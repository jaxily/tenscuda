"""Stream Hugging Face safetensors weights straight to the TT device.

``from_pretrained(...).cuda()`` first materialises the whole model in host RAM
(bf16 8B model = 16 GB; the test host has 8 GB). Here the model skeleton is built on the
``meta`` device (no memory), then each tensor is read from the mmapped
safetensors shard, cast, and moved to the card one at a time, so peak host RAM
is roughly one tensor plus the Python/torch baseline.

    from tenscuda.loader import load_causal_lm
    model = load_causal_lm("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.bfloat16)
"""
from __future__ import annotations

import glob
import json
import os
import time

import torch


def _resolve(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])


def _rebuild_nonpersistent_buffers(model, config, device):
    """Buffers like rotary ``inv_freq`` are not in the checkpoint; recompute them on CPU."""
    fixed = []
    for name, mod in model.named_modules():
        np_bufs = getattr(mod, "_non_persistent_buffers_set", set())
        if not np_bufs:
            continue
        try:
            fresh = type(mod)(config=getattr(mod, "config", config))
        except Exception as e:  # pragma: no cover - model specific
            raise NotImplementedError(f"tenscuda loader: cannot rebuild buffers of {name} "
                                      f"({type(mod).__name__}): {e}") from e
        for b in np_bufs:
            val = getattr(fresh, b)
            mod.register_buffer(b, val.to(device), persistent=False)
            fixed.append(f"{name}.{b}")
        for attr in ("original_inv_freq",):
            if hasattr(fresh, attr) and isinstance(getattr(fresh, attr), torch.Tensor):
                setattr(mod, attr, getattr(fresh, attr).to(device))
    return fixed


def load_causal_lm(model_id: str, dtype=torch.bfloat16, device=None, verbose=True):
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    import tenscuda

    device = device or tenscuda.device
    path = _resolve(model_id)
    config = AutoConfig.from_pretrained(path)
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.eval()

    params = dict(model.named_parameters(remove_duplicate=False))
    bufs = dict(model.named_buffers())
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {path}")
    loaded, nbytes = set(), 0
    import torch_xla

    for f in files:
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                target = key if key in params or key in bufs else None
                if target is None:
                    continue
                t = sf.get_tensor(key)
                if t.is_floating_point():
                    t = t.to(dtype)
                dev_t = t.to(device)
                nbytes += t.numel() * t.element_size()
                del t
                mod_name, _, leaf = target.rpartition(".")
                mod = model.get_submodule(mod_name) if mod_name else model
                if target in params:
                    mod._parameters[leaf] = torch.nn.Parameter(dev_t, requires_grad=False)
                else:
                    mod._buffers[leaf] = dev_t
                loaded.add(target)
        torch_xla.sync()
    if getattr(config, "tie_word_embeddings", False) or getattr(
            getattr(config, "get_text_config", lambda: config)(), "tie_word_embeddings", False):
        model.tie_weights()
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing:
        raise RuntimeError(f"tenscuda loader: {len(missing)} weights not in checkpoint, e.g. {missing[:5]}")
    fixed = _rebuild_nonpersistent_buffers(model, config, device)
    left = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if left:
        raise RuntimeError(f"tenscuda loader: buffers still on meta: {left[:5]}")
    try:
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(path)
    except Exception:
        pass
    torch_xla.sync()
    if verbose:
        print(f"tenscuda loader: {len(loaded)} tensors, {nbytes/2**30:.2f} GiB -> {device} "
              f"in {time.time()-t0:.1f}s; rebuilt buffers: {len(fixed)}", flush=True)
    return model


def peak_rss_gib() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def _status_gib(field: str) -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 2**20
    return float("nan")


class RssMonitor:
    """Peak *anonymous* host memory (RssAnon), sampled every ``interval`` seconds.

    VmHWM (``peak_rss_gib``) also counts file-backed pages. torch-xla keeps the CPU tensors
    handed to ``.to(device)`` alive until the plugin reads them, so the mmapped safetensors
    shards are paged in (as reclaimable page cache) during the first execution. That shows
    up in VmHWM as roughly the checkpoint size, but it's not memory the kernel can't reclaim.
    RssAnon is the number that decides whether a model fits in host RAM.
    """

    def __init__(self, interval: float = 0.01):
        import threading

        self.interval, self.peak_anon, self.peak_file = interval, 0.0, 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            self.peak_anon = max(self.peak_anon, _status_gib("RssAnon"))
            self.peak_file = max(self.peak_file, _status_gib("RssFile"))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1)
        return round(self.peak_anon, 2)
