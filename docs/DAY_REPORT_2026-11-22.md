# M4 day report — show improvement (2026-11-22)

Rock P150a. Commits on `main` (local; push only if requested).

## Headline

Opt-in `TENSCUDA_DECODE_PAUSE_MODES=1` yields **~13% lower mean decode step time** on all three decode-subset models with **8/8 kv** and **identical trace_cost tokens**.

## Results

| Model | Baseline ms | Pause ms | Δ |
|-------|-------------|----------|---|
| SmolLM2-135M | 282.1 | 243.2 | 13.8% |
| TinyLlama-1.1B | 219.3 | 190.3 | 13.2% |
| Llama-3.2-1B | 166.1 | 144.5 | 13.0% |

Artifacts: `results/m4_pause_compare_2026-11-22T181456Z.json`, `results/m4_pause_parity_gates_2026-11-22.json`.

## Why wall drops but trace_cost barely moves

`trace_cost` reports ~102 ms trace before/after pause on SmolLM2; `m4_replay_probe` wall drops ~40 ms. cProfile wall per step fell **306 → 261 ms** with pause (`m4_trace_profile_*_pause.json`). Pause exits torch-function modes during decode only; savings sit in wall accounting and non-bucketed Python overhead, not in disabling parity-critical softmax overrides on the default path.

## Copy-paste (rock)

```bash
cd ~/tenscuda
PYTHONPATH=~/tenscuda .venv/bin/python scripts/m4_pause_compare.py
PYTHONPATH=~/tenscuda TENSCUDA_DECODE_PAUSE_MODES=1 .venv/bin/python scripts/m4_replay_probe.py HuggingFaceTB/SmolLM2-135M-Instruct 32
```

## B3b

`XLA_DYNAMO_DISABLE=1` — no meaningful win (logged in PERF §9).

## Policy

Pause modes remain **opt-in** for demos/benchmarks; default generate path unchanged.