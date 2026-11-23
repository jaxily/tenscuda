# Overnight M4 unlock run — 2026-11-22/23

**Canonical learnings doc (use this going forward):** [M4_LEARNINGS.md](M4_LEARNINGS.md).

Goal: find parity-safe decode speed unlocks toward 3090-class tok/s while Jake sleeps.

## Host context (`tens`)

Fresh matrix refresh showed host upgrade already moved the needle vs legacy `rock`:

| Model | rock warm tok/s | tens warm tok/s (in-progress matrix) |
|---|---:|---:|
| SmolLM2-135M | ~3.6 | **~18** |
| Llama-3.2-1B | ~6.0 | **~19** |
| Llama-3.2-3B | ~3.5 | **~8.9** |
| Llama-3.1-8B | ~3.0 | (re-running) |

CPU still beats TT on the smallest models (~38 vs ~18 on 135M). Gap to RTX 3090 FP16 (~25–38 on 8B, hundreds on 1B) remains large.

## Runner

`scripts/overnight_m4_unlock.sh` (nohup on `tens`):

1. Wait for `run_tens_matrix_12` / `matrix.py` / `cpu_ref` to finish
2. **E5** `m4_compile_staticcache_probe.py` SmolLM2 × 16 tokens
3. `trace_cost` + `m4_replay_probe` ± pause
4. Thin-decode parity gate
5. If any compile variant matches baseline tokens → longer 32-token probe with matching env

Log: `results/overnight_m4_unlock.log`

## Code prep (opt-in only, defaults unchanged)

- `TENSCUDA_STATIC_CACHE_CPU_INIT=1` — CPU-alloc StaticCache then move buffers (upstream pattern)
- `TENSCUDA_COMPILE_LEGACY=1` — `tt_legacy_compile` for mutation-aware sync path
- Existing: `TENSCUDA_COMPILE=1`, `TENSCUDA_DECODE_PAUSE_MODES=1`, `TENSCUDA_THIN_DECODE=1`

## Success bar (honest)

| Milestone | tok/s (SmolLM2-ish) |
|---|---:|
| Started night (`tens`) | ~18 |
| Beat same-host CPU | ≳40 |
| Match 3090 FP16 small-model class | ≳100–200 |
| Match 3090 Q4 feel | ≳300+ |

Overnight target: **land at least one parity-safe compile/replay unlock** or prove remaining blockers for the upstream ask.

## Results so far (Nov 23 ~00:15)

### E5 — compile + CPU-init StaticCache
**No winners.** Best parity **2/16** (cpu_init ± legacy ± position_ids). Old compile **0/16**.
All compile variants ~1.9–2.0 s/step vs baseline tokens. Artifact:
`results/m4_compile_staticcache_SmolLM2-135M-Instruct.json`.

Device-lock lesson: parent must **not** open TT; every variant = fresh process.

### Still green
- Thin decode parity gate: **8/8 kv**, **32/32** tokens
- Warm decode on `tens` still ~**55 ms/step** (~18 tok/s) from earlier `m4_replay_probe`

### Modes probe (wave 2) — big signal

| Variant | Parity | warm ms | ≈ tok/s |
|---|---|---:|---:|
| baseline | — | 52.3 | ~19 |
| pause_only | 16/16 | **43.2** | ~23 |
| cpu_init_pause | 16/16 | **42.9** | ~23 |
| compile_pause* | **2/16** | **25.1** | ~40 wrong |

\*Always matches first **2** tokens then diverges. Compile is fast but KV/compile graph still wrong.

### Wave 3–4 — **UNLOCK**

`m4_compile_diverge.py` recipe (modes off **before** compile capture + CPU-init StaticCache):

| Model | Match | warm ms | ≈ tok/s |
|---|---|---:|---:|
| SmolLM2-135M | **32/32** | 28.1 | **~36** |
| TinyLlama-1.1B | **16/16** | 21.1 | **~47** |

Same-host CPU on 135M was ~38 tok/s — compile path now **matches CPU** on the small model.
Still short of 3090 Q4, but past “lose to CPU” for this class.

### INTEGRATED (02:42) — `TENSCUDA_COMPILE=1` works

`tenscuda/compile_unlock.py` + `static_greedy_generate` A/B
(`m4_compile_generate_ab.py`, 32 tokens):

- baseline / diverge / **generate** all **32/32** match
- warm decode **~28 ms** (~**36 tok/s**) vs eager ~53 ms (~19 tok/s)

```bash
TENSCUDA_COMPILE=1 PYTHONPATH=~/tenscuda .venv/bin/python -m tenscuda bench --model smollm2-135m
# or any path that calls static_greedy_generate
```

Still short of a 3090 Q4 stack, but **small models no longer lose to same-host CPU**.
