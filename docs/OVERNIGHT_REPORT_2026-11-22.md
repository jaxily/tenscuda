# Overnight M4 run report (2026-11-22)

Rock: `192.168.1.186` (P150a). Golden model: `HuggingFaceTB/SmolLM2-135M-Instruct`.

## Phases completed

| Phase | Outcome |
|-------|---------|
| 0 | Baselines on disk; `origin/main` was already synced earlier in the runbook |
| 1 | `m4_trace_profile.py` + E1b (prior commit `2d93275`) |
| 2 | `scripts/m4_env_sweep.py` → `results/m4_env_sweep_2026-11-22T084639Z.json`; **no** default env changes |
| 3 | `TENSCUDA_PAD_TO` + `m4_pad_bucket.py`; default **64** kept |
| 4 | B3a `decode_graph.py` + `TENSCUDA_DECODE_PAUSE_MODES=1`; kv **8/8**, ~245 vs ~281 ms replay mean (opt-in) |
| 5 | This report |

## Baselines vs end (SmolLM2-135M)

| Metric | Baseline | After overnight |
|--------|----------|-----------------|
| trace_cost trace ms | ~102 | ~102 (unchanged) |
| trace_cost sync ms | ~178 | ~176–178 |
| m4_replay mean decode ms | ~281 | ~245 with `TENSCUDA_DECODE_PAUSE_MODES=1` |
| kv_diagnose | 8/8 | 8/8 (with pause-modes env) |
| XLA compiles after warmup | 4 flat | 4 flat |

## Commits (local; not pushed unless you ask)

See `git log` since `2d93275` after this run's commits land.

## Ops notes

- **Do not** `import tenscuda` in parent processes that spawn GPU subprocesses (PJRT `CHIP_IN_USE` deadlock).
- Capture child **stdout** to files or grep JSON lines; TT-metal logs on stdout.
- One heavy `scripts/*.py` job on rock at a time.

## Next single recommendation

Smoke `TENSCUDA_DECODE_PAUSE_MODES=1` on TinyLlama + `measure_decode_subset.py`; if parity holds, document as recommended env for decode benchmarks. Continue B3b (XLA persistent graph env) only if pause-modes plateaus.