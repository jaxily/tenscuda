# Hardware baseline template

Record **one row per host** so performance comparisons stay apples-to-apples when the motherboard, CPU, or PCIe layout changes. Software SHA and harness env must match when comparing old vs new rows.

## Primary test host (upgrade in progress)

Card host upgraded **Nov 2025** (user-reported). Re-run the day-one suite below and fill in measured columns.

| Field | Value |
|-------|--------|
| Nickname | `rock` (SSH alias; same machine after rebuild) |
| CPU | **Intel Core i7-9700** (8c/8t; confirm with `lscpu` after install) |
| Host RAM | **32 GB** |
| PSU | **1200 W** |
| Motherboard | *TBD* — record model from BIOS or `dmidecode -t baseboard` |
| PCIe | *TBD* — `lspci -vv` (Gen/lanes to P150) |
| IOMMU | *TBD* |
| Card | Tenstorrent Blackhole P150a ×1 |
| OS | Ubuntu 24.04 (expected) |
| Stack | See [README](../README.md) quickstart pins |
| M4 numbers | **Pending** — run checklist below |

### Reference M4 numbers (SmolLM2-135M) — *pending on new host*

| Metric | Default path | `TENSCUDA_DECODE_PAUSE_MODES=1` |
|--------|--------------|-----------------------------------|
| `trace_cost` trace ms | — | — |
| `trace_cost` sync ms | — | — |
| `m4_replay_probe` mean decode ms | — | — |
| `m4_pause_compare` Δ% | — | — |
| kv_diagnose | — | — |

Compare against **legacy rock** row below after first suite completes.

---

## Legacy reference host (rock, pre-upgrade)

Historical measurements through **Nov 2025**. Superseded for new matrix rows once the primary host suite is committed.

| Field | Value |
|-------|--------|
| Nickname | `rock` (legacy) |
| Motherboard | H110 Pro BTC+ (unknown to UMD; tray_id fallback) |
| CPU | Intel i7-6700 (4c/8t), 2015-era |
| Host RAM | 8 GB |
| PSU | (stock) |
| PCIe | Gen3 x16 (card) |
| IOMMU | disabled (UMD log) |
| Card | Tenstorrent Blackhole P150a ×1 |

### Legacy M4 numbers (SmolLM2-135M, default prompt)

| Metric | Default path | `TENSCUDA_DECODE_PAUSE_MODES=1` |
|--------|--------------|-----------------------------------|
| `trace_cost` trace ms | ~102 | ~102 |
| `trace_cost` sync ms | ~178 | ~176 |
| `m4_replay_probe` mean decode ms | ~282 | ~243 (~**14%** faster) |
| kv_diagnose | 8/8 | 8/8 |

Artifacts: `results/m4_replay_probe_SmolLM2-135M-Instruct.json`, `results/m4_pause_compare_2026-11-22T181456Z.json`.

### Optional flags (legacy rock, Nov 2025)

| Flag | Result |
|------|--------|
| `TENSCUDA_THIN_DECODE=1` | `m4_pause_parity_gate` **8/8** kv, **32/32** tokens |
| `TENSCUDA_LOADER_PREFETCH=1` | TinyLlama stream load **1.46 s → 1.21 s** |
| `matrix.py --pause-benchmark` | SmolLM2 **14.3%** pause Δ, warm **3.56 tok/s** |

---

## Day-one checklist (primary host)

Run on the card host with `PYTHONPATH=~/tenscuda` and `.venv/bin/python` (**one script at a time**):

```bash
cd ~/tenscuda && git pull
git rev-parse HEAD   # paste into table above
.venv/bin/python scripts/trace_cost.py HuggingFaceTB/SmolLM2-135M-Instruct default
TENSCUDA_DECODE_PAUSE_MODES=1 .venv/bin/python scripts/trace_cost.py HuggingFaceTB/SmolLM2-135M-Instruct default
.venv/bin/python scripts/m4_pause_compare.py
.venv/bin/python scripts/matrix.py HuggingFaceTB/SmolLM2-135M-Instruct --pause-benchmark
.venv/bin/python scripts/loader_bench_prefetch.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Record:

- PCIe generation/lanes; whether UMD **power-state / Remote I/O** warnings disappear
- trace_ms, sync_ms, pause compare summary
- Commit updated numbers in this file and a bullet in `docs/PERF.md` §9

32 GB host RAM enables **fp32 CPU references** for more matrix models without bf16-only refs.