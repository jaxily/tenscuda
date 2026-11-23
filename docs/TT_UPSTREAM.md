# Tenstorrent stack — upstream requests and E4 hooks

TensCUDA stays a thin device adapter; **graph replay**, **device DRAM introspection**, and **native op routing** depend on pjrt-plugin-tt / TT-XLA guidance.

## Blockers in-repo (Nov 2026)

| Need | Symptom | TensCUDA workaround | Upstream ask |
|------|---------|---------------------|--------------|
| Device memory stats | `xm.get_memory_info` UNIMPLEMENTED | Cannot verify bfp8 DRAM savings (PERF §7) | Implement `PJRT_Device_MemoryStats` or equivalent on PJRT path |
| Graph replay for decode | ~100 ms/step Python re-trace; wall wins from pause modes only | Per-step sync + StaticCache; `decode_graph.py`, B3b env sweeps | Document sanctioned **trace capture / persistent xla graph** API for HF-style loops |
| Fused softmax accuracy | ~2e-2 rel L2 | `accurate_ops` reciprocal softmax | Fix fused op or expose accurate SDPA in compiler |
| StaticCache + multi-forward | Wrong tokens if sync deferred | One sync per decode forward | Functional KV / replay safe with batched forwards (if ever) |

## E4 native op policy (this repo)

1. **No hand-written tt-metal C++ kernels** in TensCUDA.
2. Route hot ops only via **documented** TT-XLA hooks (custom call, op override, tt_torch extension) with `scripts/op_parity.py` + `m4_pause_parity_gate.py` gates.
3. Candidates by profile: **SDPA**, **RMSNorm**, **rotary** — measure decode delta on SmolLM2-135M before matrix-wide claims.
4. **Landed (Nov 23):** `TENSCUDA_NATIVE_OPS=rms_norm|sdpa,rms_norm` → `tt_torch.composite_ops` in
   `tenscuda/native_ops.py`. A/B: `scripts/m4_native_compile_ab.py` (**32/32**, no 8B ms win;
   stay opt-in). Eager-only gate: `scripts/m4_e4_generate_ab.py`.

## Research log (bounded)

| Item | Result | Reference |
|------|--------|-----------|
| `XLA_DYNAMO_DISABLE=1` | No meaningful trace/sync win; tokens OK | PERF §9 B3b |
| `TENSCUDA_DECODE_PAUSE_MODES=1` | ~13% wall ms/step, 3/3 subset parity | `docs/DAY_REPORT_2026-11-22.md` |
| `TENSCUDA_THIN_DECODE=1` | Calls `model.forward` directly; measure on rock | `decode_graph.get_forward_fn` |
| Multi-chip (M7) | `xr.global_runtime_device_count()` probed; single-device path only | VISION M7 |
| **Compile / graph replay (Nov 22 research)** | See below | `scripts/m4_compile_staticcache_probe.py` |

### Compile / no-retrace path (Nov 22)

**Installed on `tens`:** `pjrt-plugin-tt 1.5.0.dev20260831` already ships `extract_compiled_graph` and
`tt_legacy_compile` / experimental compile (experimental is **default** in current `tt_torch` backend).

**What upstream does that we do not (yet):**

1. **CPU-init `StaticCache`** then move `keys` / `values` / `cumulative_length` to XLA.
   Device-side construction traces unfusable creation ops (`tests/torch/models/llama3/test_llama_step_n300.py`).
2. **`torch.compile(backend="tt")`** with the experimental executor → `bridge.extract_compiled_graph(...)`
   so later forwards skip Python re-trace (the ~trace-ms half of our step).
3. Legacy path still has an explicit branch for **`USER_INPUT_MUTATION`** (inplace StaticCache
   updates) via `torch_xla.sync()` when `has_output_mutations`; experimental path relies on
   `extract_compiled_graph` + caller sync.
4. Their llama step test uses **`position_ids`**, not only `cache_position`.
5. Separate upstream work: Dynamo KV identity guards / sliding-window Python ints cause
   per-token recompile ([tt-xla#4444](https://github.com/tenstorrent/tt-xla/issues/4444), sliding-window PR).
   Metal trace can win ~2× on tiny models but RoPE/`from_device` still blocks Llama
   ([tt-xla#4220](https://github.com/tenstorrent/tt-xla/issues/4220)).

**Historical failure (modes-on capture):** wrong from token 1 / classic **2/16** diverge.
**Unlock (Nov 23):** modes **off** before Dynamo + CPU-init `StaticCache` + `torch.compile(backend="tt")`
→ SmolLM2 **32/32** @ ~36 tok/s (`TENSCUDA_COMPILE=1`). Recipe: [M4_LEARNINGS.md](M4_LEARNINGS.md).

**Still do not default-on** until CPU teacher gates pass on more models. Remaining TensCUDA levers:
E4 native ttnn, plugin/`TTIRConsolidateStaticCacheUpdates` check, NVIDIA same-harness ref.

**Related upstream perf (not yet claimed in TensCUDA):**
- tt-mlir `TTIRConsolidateStaticCacheUpdates` recovered most of a **~27%** transformers-5.5
  decode regression from per-layer `cumulative_length` adds
  ([tt-mlir#8277](https://github.com/tenstorrent/tt-mlir/commit/2d90ef04f303237e19aef629b86ecf296b5d98da)).
  Confirm whether our Aug-2026 pjrt wheel includes that pass; if not, upgrading the plugin is an
  M4 lever independent of TensCUDA Python.
- HF `torch.export` decode runner models StaticCache as `USER_INPUT_MUTATION` (zero-copy in-place);
  worth a future spike once compile parity lands.

## How to engage Tenstorrent

- Attach **minimal repro**: model id, `scripts/kv_diagnose.py` output, `trace_cost` JSON, commit SHA.
- Reference **compatibility matrix** row + `results/` JSON.
- Ask explicitly for: (1) decode graph persistence, (2) PJRT memory view, (3) custom-call example for SDPA.

File issues on [tt-xla](https://github.com/tenstorrent/tt-xla) / pjrt-plugin-tt with links to this doc when opening upstream tickets.