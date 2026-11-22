# Hardware baseline template

Record **one row per host** so performance comparisons stay apples-to-apples when the motherboard, CPU, or PCIe layout changes. Software SHA and harness env must match when comparing old vs new rows.

## Current reference host (rock)

| Field | Value |
|-------|--------|
| Nickname | `rock` |
| Motherboard | H110 Pro BTC+ (unknown to UMD; tray_id fallback) |
| CPU | Intel i7-6700 (4c/8t), 2015-era |
| Host RAM | 8 GB |
| PCIe | Gen3 x16 (card) |
| IOMMU | disabled (UMD log) |
| Card | Tenstorrent Blackhole P150a ×1 |
| OS | Ubuntu 24.04 (typical) |
| KMD | 2.11.0 |
| Firmware | bundle 18.10.0 |
| pjrt-plugin-tt | 1.5.0.dev20260831 |
| torch / torch-xla | 2.11.0+cpu / 2.9.0 (TT build) |
| TensCUDA commit | `git rev-parse HEAD` at measure time |

### Reference M4 numbers (SmolLM2-135M, default prompt)

| Metric | Default path | `TENSCUDA_DECODE_PAUSE_MODES=1` |
|--------|--------------|-----------------------------------|
| `trace_cost` trace ms | ~102 | ~102 |
| `trace_cost` sync ms | ~178 | ~176 |
| `m4_replay_probe` mean decode ms | ~282 | ~243 |
| kv_diagnose | 8/8 | 8/8 |

Artifacts: `results/m4_replay_probe_SmolLM2-135M-Instruct.json`, `results/m4_pause_compare_2026-11-22T181456Z.json`.

## New motherboard — day-one checklist

Run on the card host with `PYTHONPATH=~/tenscuda` and `.venv/bin/python`:

```bash
git pull   # pin commit; record SHA below
.venv/bin/python scripts/trace_cost.py HuggingFaceTB/SmolLM2-135M-Instruct default
TENSCUDA_DECODE_PAUSE_MODES=1 .venv/bin/python scripts/trace_cost.py HuggingFaceTB/SmolLM2-135M-Instruct default
.venv/bin/python scripts/m4_pause_compare.py HuggingFaceTB/SmolLM2-135M-Instruct
.venv/bin/python scripts/matrix.py HuggingFaceTB/SmolLM2-135M-Instruct --pause-benchmark
```

Add a **new table row** above with:

- PCIe generation/lanes (and whether UMD power-state warnings disappear)
- IOMMU on/off if you test both
- trace_ms, sync_ms, pause compare summary
- Optional: one A/B with `TENSCUDA_LOADER_PREFETCH=1` on a 1–2B model (load time only)

Update `docs/PERF.md` §9 with a short “host comparison” bullet linking this file.

## New host (blank row)

| Field | Value |
|-------|--------|
| Nickname | |
| Motherboard | |
| CPU | |
| Host RAM | |
| PCIe | |
| IOMMU | |
| Measured at commit | |
| trace / sync / replay (default) | |
| replay (pause) | |
| Notes | |