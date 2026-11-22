# Compile cache and shape buckets (stable across hosts)

Tenstorrent decode performance depends on **reusing** the same TT-MLIR executables. TensCUDA keeps shapes fixed; you keep the **JIT cache** warm.

## tt-metal JIT cache

- Default location: `~/.cache/tt-metal-cache` on the card host.
- **Do not delete** this directory between benchmark runs; first run after wipe pays full kernel JIT cost (logs: `BuildKernels | JIT cache stats`).
- Matrix and `m4_replay_probe` numbers assume a **warmed** cache unless labeled “cold / first run”.

## XLA compile graphs (typical Llama-like decode)

- **Prefill** (prompt length N) + **decode** (1×1 token, fixed `StaticCache` max length) → usually **4** compile events plateauing after step 1 (`m4_replay_probe` `xla_compiles_per_decode_step`).
- **`first run s`** in the matrix is dominated by compile; **`warm tok/s`** is runs 2–4 only.

## Shape buckets (`TENSCUDA_PAD_TO`)

StaticCache `max_cache_len` is rounded up to a multiple of `pad_to` ([`tenscuda/generate.py`](../tenscuda/generate.py)).

Standard matrix / bench prompt ≈ **46 tokens** + **32** new tokens → need `max_cache_len ≥ 78`.

| `pad_to` | `max_cache_len` (46+32) | E3 first-run (135M) | Notes |
|----------|-------------------------|---------------------|--------|
| **64** | 64 | ~45.5 s | **Default** — best first-run for this prompt length |
| 128 | 128 | ~45.8 s | Same compile count; slightly larger cache tensor |
| 256 | 256 | ~56.2 s | Slower first run; use only for longer `max_new_tokens` |

Set only when prompts are longer: `TENSCUDA_PAD_TO=128` (or 256).

## Program cache on device

PJRT/tt-metal logs `Enabling program cache on MeshDevice` at startup. Keeping **input shapes stable** (static greedy generate, fixed decode tensors) maximizes hits.

## Benchmark checklist

1. One heavy `scripts/*.py` job at a time on the card host.
2. Warm JIT cache before timing (one throwaway generate).
3. Record **default** and **`TENSCUDA_DECODE_PAUSE_MODES=1`** separately when reporting M4 wins.
4. After a motherboard swap, re-run `trace_cost` + `m4_pause_compare` — see [HARDWARE_BASELINE.md](HARDWARE_BASELINE.md).

## Loader CPU prefetch (host threads only)

``TENSCUDA_LOADER_PREFETCH=1`` overlaps **CPU** safetensors reads with **H2D + sync** on the main thread ([`tenscuda/loader.py`](../tenscuda/loader.py)). Does not touch decode. Measure with:

```bash
python scripts/loader_bench_prefetch.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
```