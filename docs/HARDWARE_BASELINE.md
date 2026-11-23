# Hardware baseline template

Record **one row per host** so performance comparisons stay apples-to-apples when the motherboard, CPU, or PCIe layout changes. Software SHA and harness env must match when comparing old vs new rows.

## Primary test host (`tens`)

SSH alias **`tens`** (192.168.1.210). Card host rebuilt **Nov 2025** with Ryzen 9700X; KMD + venv SFPI 7.69.0 on **2026-11-22**.

| Field | Value |
|-------|--------|
| Nickname | `tens` |
| CPU | **AMD Ryzen 7 9700X** (8c/16t) |
| Host RAM | **~30 GiB** |
| PSU | (stock) |
| Motherboard | **Gigabyte B850 GAMING X WIFI6E** (UMD tray_id fallback; bus 0x1) |
| PCIe | *pending `sudo lspci -vv`* — card at `01:00.0`, BusMaster+ after KMD |
| IOMMU | **enabled** (UMD log) |
| Secure Boot | **disabled** (required for unsigned `tenstorrent` DKMS) |
| Card | Tenstorrent Blackhole P150a ×1, firmware bundle **18.10.0** |
| OS | Ubuntu 24.04, kernel **7.0.0-43-generic** |
| KMD | **2.11.0** (`tenstorrent-dkms`); device **`/dev/tenstorrent/0`** (not `tenstorrent0`) |
| Stack | See [README](../README.md) quickstart pins |
| TensCUDA commit | `a4a9d26` (rsync from dev; host has no `git` yet) |
| M4 numbers | **2026-11-22** suite on `tens` (see below) |

UMD notes (non-fatal): `Setting power state failed: Remote I/O error`; eth FW 1.6.0 → single-erisc fallback.

### Reference M4 numbers (SmolLM2-135M) — measured on `tens` (2026-11-22, re-checked 2026-11-23)

| Metric | Default path | `TENSCUDA_DECODE_PAUSE_MODES=1` |
|--------|--------------|-----------------------------------|
| `trace_cost` trace ms | **22.7** (23-Nov: **21.9**) | **22.5** |
| `trace_cost` sync ms | **32.9** (23-Nov: **32.2**) | **33.6** |
| `m4_replay_probe` mean decode ms | **53.1** | **43.7** (**17.7%** faster; 22-Nov suite ~55 / ~45) |
| `m4_pause_compare` Δ% (3 models) | SmolLM2 **18.9%**, TinyLlama **9.3%**, Llama-3.2-1B **7.4%** | |
| `measure_decode_subset` warm tok/s | SmolLM2 **~18**, TinyLlama **~17.8**, Llama-3.2-1B **~19.2** | |
| kv_diagnose | **8/8** | **8/8** |

Artifacts: `results/tens_host/m4_*`, `results/tens_host/tens_m4_suite.log` (synced from `tens`).

Compare against **legacy rock** (i7-6700 / 8 GB): trace ~102 / sync ~178 ms, replay ~282 / ~243 ms, warm ~3.5 tok/s — **host upgrade dominates decode wall time on SmolLM2.**

### Optional flags (`tens`, 2026-11-22)

| Flag | Result |
|------|--------|
| `TENSCUDA_DECODE_PAUSE_MODES=1` | SmolLM2 replay **18.9%** faster (env sweep replay **53.4 → 43.2 ms**) |
| `TENSCUDA_THIN_DECODE=1` | **8/8** kv, **32/32** tokens |
| `TENSCUDA_LOADER_PREFETCH=1` | TinyLlama stream load **0.66 s → 0.59 s** |

Full report: [BENCHMARK_REPORT_tens_2026-11-22.md](BENCHMARK_REPORT_tens_2026-11-22.md).

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
ls -l /dev/tenstorrent/0 && PYTHONPATH=. .venv/bin/python -c "import tenscuda; print(tenscuda.is_active)"
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