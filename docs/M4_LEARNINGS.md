# M4 learnings — keep optimizing from here

**Last update:** 2026-11-23 (overnight unlock on `tens`).  
**Status:** compile unlock **default on** (`TENSCUDA_COMPILE=1`; set `=0` for eager). Still far
from metal batch-1 (~40 tok/s) and 3090-class.

Canonical code: [`tenscuda/compile_unlock.py`](../tenscuda/compile_unlock.py)  
Wired through: [`tenscuda/generate.py`](../tenscuda/generate.py) when `TENSCUDA_COMPILE=1`
(or `scripts/matrix.py MODEL --compile-bench` for matrix rows).  
Experiment log: [`PERF.md`](PERF.md) §9 · Queue: [`M4.md`](M4.md) · Upstream: [`TT_UPSTREAM.md`](TT_UPSTREAM.md)

---

## Speed ladder (SmolLM2-135M, batch 1, greedy, `tens` P150a)

| Stage | Warm decode | ≈ tok/s | Parity |
|---|---:|---:|---|
| Legacy `rock` eager | ~280 ms | ~3.6 | 32/32 vs CPU |
| `tens` eager (host upgrade) | ~53–55 ms | ~18–19 | 32/32 |
| + `TENSCUDA_DECODE_PAUSE_MODES=1` | ~43 ms | ~23 | 32/32 |
| + **`TENSCUDA_COMPILE=1`** (unlock) | **~28 ms** | **~36** | **32/32 vs eager** |
| Llama-3.1-8B (`tens`, Nov 23) | eager **~206 ms** → compile **~90 ms** | **~5 → ~11** | **32/32** (`results/m4_compile_generate_ab_8b.json`) |
| Same-host CPU fp32 | — | ~38 | reference |
| RTX 3090 FP16 (public est.) | — | ~200–400* | not same harness |
| RTX 3090 Q4 llama.cpp (public) | — | ~400+* | not same harness |

\*Web estimates — not `scripts/matrix.py` on the same prompt.

**Other unlock measurements (`tens`):** TinyLlama ~**47 tok/s** (16/16); Llama-3.2-1B ~**43 tok/s** (16/16).

Artifacts: `results/m4_compile_generate_ab.json`, `results/m4_compile_diverge_*.json`, `results/m4_compile_modes_*.json`.

---

## What actually worked (do not regress)

### 1. Host upgrade matters as much as software (so far)
Moving from i7-6700 / 8 GB (`rock`) to Ryzen 7 9700X / 30 GB (`tens`) cut trace+sync from ~280 ms/step to ~55 ms **before** any compile unlock. Always report which host.

### 2. Pause modes (eager path) — safe ~15–20%
`TENSCUDA_DECODE_PAUSE_MODES=1` exits TensCUDA + tt_torch `__torch_function__` modes during **decode only**. Validated 3/3 subset, 8/8 kv. Does **not** remove re-trace; trims mode overhead.

### 3. Compile unlock — the real step change
Recipe that hit parity + ~2× vs `tens` eager:

1. **CPU-init `StaticCache`** (`device="cpu"`, dtype bf16) then `early_initialization(...)` then `.to(xla)` for keys/values/`cumulative_length`  
   — device-side construction traces unfusable creation ops (tt-xla llama tests).
2. **Exit torch-function modes before Dynamo capture** and keep them off for the whole generate  
   — capturing with modes on → wrong tokens (classic **2/16** diverge after two matching tokens).
3. **`torch.compile(backend="tt")`** (experimental / `extract_compiled_graph` is default in our wheel).
4. **One `torch_xla.sync()` per decode forward** (StaticCache still requires this).
5. **Argmax via `logits.float()` + topk** in the unlock loop (near-ties are common; bf16 argmax can flip).

```bash
# Reproduce unlock A/B (fresh process per variant)
PYTHONPATH=~/tenscuda .venv/bin/python scripts/m4_compile_generate_ab.py \
  HuggingFaceTB/SmolLM2-135M-Instruct 32

# Use in generate / bench
TENSCUDA_COMPILE=1 PYTHONPATH=~/tenscuda .venv/bin/python scripts/m4_replay_probe.py \
  HuggingFaceTB/SmolLM2-135M-Instruct 32
```

Default path stays eager (modes on) until CPU-teacher gates pass on more models.

---

## What failed (do not retry blindly)

| Attempt | Result | Lesson |
|---|---|---|
| Multi-forward lazy graph (K>1 sync) | 3/8 kv | One sync per decode forward forever |
| `TENSCUDA_RESIDENT_DECODE_IO=1` | +1 compile/token, ~21 s/step | In-place step buffer mutation kills reuse |
| `torch.compile` with modes **on** | wrong from token 1 / **2/16** | Capture must see modes exited |
| CPU-init alone + compile (modes on) | still **2/16** | CPU-init necessary but not sufficient |
| `tt_legacy_compile` alone | no unlock | Mutation sync path ≠ modes-off capture |
| `position_ids` vs `cache_position` | no unlock by itself | Not the diverge root |
| Inlining unlock loop into `generate.py` | A/B stayed **2/16** | Keep body in `compile_unlock.py`; call it — do not “simplify” casually |
| Parent process holding TT while spawning variants | child hung on `CHIP_IN_USE` | One process per TT job; parent must not open device |

---

## Hard constraints (still true)

1. HF `StaticCache` + deferred sync across decode forwards → wrong tokens (`kv_diagnose`).
2. Eager path still needs TensCUDA softmax/SDPA overrides for parity (`trace_cost no_modes`).
3. Compile unlock runs **modes-off** for the compiled graph; greedy top-1 matched eager on 135M/1.1B/1B overnight — **still need CPU teacher gates** before default-on.
4. One heavy TT job at a time on the card host.

---

## Where the remaining time is

On `tens` eager, `trace_cost` was ~**23 ms trace / 33 ms sync**. Compile removes most Python re-trace; warm ~28 ms is largely **device execute + sync + leftover host**.

Next levers (priority order):

| Next | Why | How to measure |
|---|---|---|
| **CPU teacher gate** on compile path (135M → 3B → 8B) | Unlock must not be “eager-matched but CPU-wrong” | extend `m4_pause_parity_gate` / matrix with `TENSCUDA_COMPILE=1` |
| **E4 native ttnn** (SDPA / RMSNorm / rotary) | Shrink sync+exec half | `op_parity.py` + decode A/B on 135M |
| **Plugin / tt-mlir upgrade** | `TTIRConsolidateStaticCacheUpdates` recovered ~27% upstream once | compare pjrt wheel vs May-2026 pass; re-run probe |
| **NVIDIA same-harness ref** | Honest 3090 FP16 gap | `results/nvidia_ref_*.json` with matrix prompt |
| **Default-on decision** | Users shouldn’t need the env var if safe | only after teacher gates + matrix subset |

Out of scope for TensCUDA (point upstream): metal trace RoPE/`from_device` blocks ([tt-xla#4220](https://github.com/tenstorrent/tt-xla/issues/4220)); sanctioned persistent decode API docs ([TT_UPSTREAM.md](TT_UPSTREAM.md)).

---

## Env cheat sheet

| Env | Default | Effect |
|---|---|---|
| `TENSCUDA_COMPILE` | **1** (default) | Compile unlock (`compile_unlock.py`); **0** = eager |
| `TENSCUDA_NATIVE_OPS` | off | `sdpa,rms_norm` etc.; A/B 32/32, **no 8B ms win** Nov 23 |
| `TENSCUDA_NATIVE_DEFAULT` | off | Set `1` to auto-fill `TENSCUDA_NATIVE_OPS` on import |
| `TENSCUDA_DECODE_PAUSE_MODES=1` | off | Eager decode: pause modes (also set by unlock) |
| `TENSCUDA_STATIC_CACHE_CPU_INIT=1` | off | CPU-init cache (forced by unlock) |
| `TENSCUDA_THIN_DECODE=1` | off | `model.forward` bind; parity OK, small win |
| `TENSCUDA_COMPILE_LEGACY=1` | off | `tt_legacy_compile`; no unlock alone |
| `TENSCUDA_RESIDENT_DECODE_IO=1` | off | **Broken** — do not use |

---

## Overnight narrative (archive)

Full chronological notes: [OVERNIGHT_UNLOCK_2026-11-22.md](OVERNIGHT_UNLOCK_2026-11-22.md).  
Day pause report: [DAY_REPORT_2026-11-22.md](DAY_REPORT_2026-11-22.md).
