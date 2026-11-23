# Benchmark report — host `tens` (2026-11-22)

Primary card host after Ryzen 9700X upgrade. Compare to legacy **`rock`** (i7-6700 / 8 GiB) in [HARDWARE_BASELINE.md](HARDWARE_BASELINE.md). **NVIDIA CUDA:** not measured this run (see [Future reference](#future-nvidia-reference)).

## Host

| Field | Value |
|-------|--------|
| SSH | `jake@tens` |
| CPU | AMD Ryzen 7 9700X (8c/16t) |
| RAM | ~30 GiB |
| Board | Gigabyte B850 GAMING X WIFI6E |
| OS / kernel | Ubuntu 24.04, 7.0.0-43-generic |
| KMD | 2.11.0 (Secure Boot **off**) |
| Card | Blackhole P150a, FW bundle 18.10.0 |
| TensCUDA | `a4a9d26` (+ local harness scripts, rsync install) |

Bring-up: DKMS install, **Secure Boot off**, `modprobe tenstorrent`, venv-local **SFPI 7.69.0** (see README).
Smoke: **`/dev/tenstorrent/0`** (KMD 2.11+), `tenscuda.is_active` → `True`. UMD warnings (power-state, B850 tray_id) are benign on this board.

### Re-verified 2026-11-23 (manual on `tens`)

| Check | Result |
|-------|--------|
| `trace_cost.py` default | trace **21.9 ms**, sync **32.2 ms** |
| `m4_replay_probe` 32 steps | default **53.1 ms/step**, pause **43.7 ms/step** → **17.7%** Δ |
| Artifacts | `results/m4_replay_probe_SmolLM2-135M-Instruct.json` (+ `_pause.json`) |

## M4 headline (SmolLM2-135M)

| Metric | legacy `rock` | **`tens`** | Notes |
|--------|---------------|------------|--------|
| trace_ms | ~102 | **22.7** | `trace_cost.py` |
| sync_ms | ~178 | **32.9** | |
| replay mean decode ms | ~282 | **53.1** | pause **43.7** (**17.7%** Δ, 2026-11-23) |
| HF matrix warm tok/s | ~3.55 | **~18** | first compile **~8–10 s** vs ~50 s |
| CPU ref tok/s (fp32) | ~15 | **~38** | same harness |

Subset functional decode (`measure_decode_subset_2026-11-23T070048Z.json`): SmolLM2 **17.95**, TinyLlama **17.81**, Llama-3.2-1B **19.24** tok/s.

## Matrix tier-1 (12 models)

Re-run 2026-11-22 on `tens` (~31 min wall, incl. Gemma-2-2b / Phi-3.5). Table: [COMPATIBILITY.md](COMPATIBILITY.md). Highlights vs legacy `rock` matrix:

| Model | Old warm tok/s | `tens` warm tok/s | Old first run s | `tens` first run s |
|--------|----------------|-------------------|-----------------|-------------------|
| SmolLM2-135M | 3.55 | **18.0** | 50 | **8** |
| TinyLlama-1.1B | 4.53 | **16.6** | 33 | **11** |
| Gemma-2-2b-it | 2.27 | **7.94** | 792 | **137** |
| Phi-3.5-mini | 2.67 | **6.89** | 640 | **110** |
| Llama-3.1-8B | 3.00 | **4.88** | 198 | **29** |

Many models now use **fp32 CPU refs** (30 GiB host). Phi near-tie at step 6 preserved (margin 0.036).

## Tune (opt-in flags, `tens`)

| Flag | Result |
|------|--------|
| `TENSCUDA_DECODE_PAUSE_MODES=1` | SmolLM2 **18.9%** replay Δ (see `m4_pause_compare_2026-11-23T070017Z.json`) |
| `TENSCUDA_THIN_DECODE=1` | **8/8** kv, **32/32** tokens (`m4_pause_parity_gate`) |
| `TENSCUDA_LOADER_PREFETCH=1` | TinyLlama load **0.66 s → 0.59 s** |
| `m4_env_sweep` | `decode_pause_modes` best replay win; **hifi2** / **lofi** change tokens (do not default) |

Artifacts: `results/tens_host/`, `results/matrix/` (synced from `tens`), logs `results/tens_host/tens_m4_suite.log`, `tens_matrix_12.log`, `tens_tune.log`.

## Gates

- `pytest` on `tens`: **31 passed** (fix: hostname `tens` vs substring in `tenscuda` in scrub test)
- `op_parity.py`: OK, 27 XLA compiles
- SmolLM2 `kv_diagnose`: **8/8**

## Future: NVIDIA reference

When a CUDA box is available:

1. Same models, chat prompt, 32 greedy tokens (matrix harness).
2. Native `device=cuda` (no `import tenscuda`).
3. Store `results/nvidia_ref_<model>_<date>.json`.
4. Compare columns: **TT `tens` warm tok/s** vs **CUDA warm tok/s** vs **CPU tok/s** on each host.

Optional follow-up: `scripts/nvidia_ref_baseline.py` (not implemented yet).