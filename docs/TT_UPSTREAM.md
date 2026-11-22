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

## Research log (bounded)

| Item | Result | Reference |
|------|--------|-----------|
| `XLA_DYNAMO_DISABLE=1` | No meaningful trace/sync win; tokens OK | PERF §9 B3b |
| `TENSCUDA_DECODE_PAUSE_MODES=1` | ~13% wall ms/step, 3/3 subset parity | `docs/DAY_REPORT_2026-11-22.md` |
| `TENSCUDA_THIN_DECODE=1` | Calls `model.forward` directly; measure on rock | `decode_graph.get_forward_fn` |
| Multi-chip (M7) | `xr.global_runtime_device_count()` probed; single-device path only | VISION M7 |

## How to engage Tenstorrent

- Attach **minimal repro**: model id, `scripts/kv_diagnose.py` output, `trace_cost` JSON, commit SHA.
- Reference **compatibility matrix** row + `results/` JSON.
- Ask explicitly for: (1) decode graph persistence, (2) PJRT memory view, (3) custom-call example for SDPA.

File issues on [tt-xla](https://github.com/tenstorrent/tt-xla) / pjrt-plugin-tt with links to this doc when opening upstream tickets.