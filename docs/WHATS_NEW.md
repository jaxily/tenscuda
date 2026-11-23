# What's new (Nov 23, 2026)

Highlights for reviewers and demos — all numbers have committed JSON under `results/`.

## Native speed (pitch this first)

On **one P150** (`tens`), **Llama-3.1-8B**:

| Stack | Throughput | vs CPU (~3 tok/s) |
|---|---:|---:|
| tt-metal batch-32 | **~1079 tok/s** | **~354×** |
| vLLM-TT offline (32 seq) | **~909 tok/s** | **~300×** |
| TensCUDA compile (batch 1) | **~11 tok/s** | **~4×** |

One command: `bash scripts/run_tt_speed_showcase.sh` — docs in [SHOWCASE_CUDA.md](SHOWCASE_CUDA.md).

## TensCUDA compatibility

- **15 models** 135M–**9B** in [COMPATIBILITY.md](COMPATIBILITY.md); **Gemma-2-9B** largest proved row (**3.81 tok/s**, 32/32).
- **8B:** 32/32 parity, streaming load on 8 GB host.
- Unified compat vs speed: [TEST_RESULTS.md](TEST_RESULTS.md).

## Code / defaults

- `TENSCUDA_COMPILE=1` default for generate; `=0` to force eager.
- Showcase harness: `run_tt_speed_showcase.sh`, `parse_tt_speed_logs.py`, `render_test_results.py`.
- E4 native ops stay opt-in (`TENSCUDA_NATIVE_OPS` / `TENSCUDA_NATIVE_DEFAULT=1`).

## Artifacts to open in git

| File | What |
|---|---|
| [TEST_RESULTS.md](TEST_RESULTS.md) | Summary tables + artifact index |
| `results/showcase_2026-11-23T2151Z.json` | Latest metal + vLLM bundle |
| `results/vllm_offline_b32_speed_2026-11-23T2222.json` | Latest vLLM offline parse |
| `results/tt_metal_b32_speed_2026-11-23T1309.json` | Metal pytest batch-32 |
| `results/m4_native_compile_ab_*.json` | Compile vs native-op A/B |