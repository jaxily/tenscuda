# Contributing to TensCUDA

Thanks for helping make Tenstorrent cards easier to use from PyTorch.

## Reference measurement host

Most committed matrix numbers come from one Blackhole P150a on an 8 GB Ubuntu box. From the
maintainer laptop, runs are started over SSH (interactive TTY helps on long compiles):

```bash
ssh -t rock 'cd ~/tenscuda && PYTHONPATH=~/tenscuda .venv/bin/python scripts/matrix.py <hf-model-id>'
```

On **rock** itself:

```bash
cd ~/tenscuda
export PYTHONPATH=~/tenscuda:$PYTHONPATH
.venv/bin/python scripts/matrix.py <hf-model-id> --new-tokens 32
.venv/bin/python scripts/matrix.py --render   # optional; matrix also re-renders after each model
```

**8B models on an 8 GB host** cannot load a full CPU bf16 reference in RAM. Skip CPU and still
record TT load, decode, and RSS:

```bash
.venv/bin/python scripts/matrix.py unsloth/Meta-Llama-3.1-8B-Instruct --cpu-dtype skip
```

**Gated Hugging Face checkpoints** (Meta Llama, Google Gemma instruct, etc.): use an ungated mirror
with the same architecture when the host token lacks access, and note it in the matrix footer.
Examples already in the matrix: `unsloth/Llama-3.2-*`, `unsloth/gemma-2-2b-it` for
`google/gemma-2-2b-it`.

**Disk and downloads:** Large `snapshot_download` / Xet failures often mean the disk is full.
Check `df -h /` and `du -sh ~/.cache/huggingface/hub/models--* | sort -h`. Remove abandoned model
caches before new 5–15 GB downloads. If downloads stall with Xet writer errors, try:

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=~/tenscuda .venv/bin/python scripts/matrix.py ...
```

**Logs:** `results/matrix/<short>_cpu.log`, `results/matrix/<short>_tt.log`, and
`results/matrix/<short>.json` (combined row). TT step output goes to
`results/load_<short>_stream.json`.

**One heavy job at a time on the 8 GB host.** Do not start a second `matrix.py`, `cpu_ref.py`,
or `load_bench.py` while another 8B (or large mmap'd CPU ref) run is in progress. The low-memory
watchdog in `scripts/matrix.py` can kill subprocesses when `MemAvailable` drops, and overlapping
loads were seen to OOM-kill new jobs immediately. Typical flow for Llama-3.1-8B:

```bash
# 1) CPU reference alone (~15–20 min, mmap bf16)
.venv/bin/python scripts/cpu_ref.py unsloth/Meta-Llama-3.1-8B-Instruct bfloat16 32 \
  results/matrix/Meta-Llama-3.1-8B-Instruct_cpu_bfloat16.json

# 2) After it finishes, full matrix (reuses CPU JSON + fresh TT), or TT only:
export TENSCUDA_TEACHER=results/matrix/Meta-Llama-3.1-8B-Instruct_cpu_bfloat16.json
.venv/bin/python -u scripts/load_bench.py unsloth/Meta-Llama-3.1-8B-Instruct stream 32
```

Or a single `matrix.py … --cpu-dtype bfloat16` when nothing else is using the card or RAM.

## Model compatibility reports

```bash
python scripts/matrix.py <hf-model-id> --new-tokens 32
```

This runs a CPU reference when it fits in RAM, then a streaming TT run (one first run + three warm
runs), and appends a row to `docs/COMPATIBILITY.md`. Please review the row, attach the JSON from
`results/`, and open a PR or use the **Model compatibility report** issue template.

Report what actually happened. "Works" means the script runs and its output agrees with the CPU
reference. If tokens differ, include the matching prefix length and logits/hidden-state error if
you can, not just "it generated text". Any quantized variant must run in a **fresh process** (the
TT-XLA graph cache may reuse the first weight dtype).

## Benchmark results

```bash
python -m tenscuda bench --model <key>          # see --list; add --quick for a short run
```

Review the printed JSON, then either paste it into a **Benchmark result** issue or open a PR adding
the file, unedited, to `benchmarks/results/` (name it `<model>-<card>-<date>.json`). Run
`python scripts/build_leaderboard.py` to regenerate `docs/LEADERBOARD.md`; `--check` validates
the files. Please submit only runs without accelerator faults or thermal warnings, with no other
jobs on the card. Note anything non-default in the issue.

## Bug reports

Include the card and host specs, OS, KMD/firmware versions, Python and torch/torch-xla/pjrt-plugin-tt
versions, a minimal script, and the complete error. Scrub keys, tokens, IP addresses, hostnames and
private file paths from logs before posting them.

## Code changes

- Use an isolated venv. Don't change a working system's driver, firmware or existing app environment.
- Keep changes small, with a test or reproduction and a measured result.
- Run `python -m pytest -q tests/` on a Tenstorrent card. There's no hosted CI yet.
- Run op parity tests (`python scripts/op_parity.py`) when changing numerical code.
- For speed changes, include first-run time separately from warm speed, at least 3 warm repeats,
  CPU token parity (or explicit error metrics), peak host RSS and any CPU fallback ops.
- Log what you learn in `docs/PERF.md`, including failed experiments. Keep only parity-preserving
  optimizations on the default path.
- Do not commit weights, tensor caches, compiled kernels, model outputs with private prompts, keys,
  or credentials. `.gitignore` covers the common cases.

## Milestone status (Nov 2026)

| Milestone | Status | What’s left |
|---|---|---|
| **M1** Tier-1 matrix | Done | — |
| **M2** 3–4B (Llama-3.2-3B, Qwen2.5-3B, Phi-3.5-mini, Gemma-2-2B) | Done | Phi: greedy near-tie (teacher 31/32); document only |
| **M3** 8B small host | Done | 32/32 vs mmap'd CPU bf16 on 8 GB host; TT anon **0.52 GiB** |
| **M4** Graph replay + native kernels | In progress | **3/3** subset ~13% decode step win with opt-in `TENSCUDA_DECODE_PAUSE_MODES=1` (8/8 kv, 32/32 tokens); run `scripts/m4_pause_compare.py` — see `docs/DAY_REPORT_2026-11-22.md` |
| **M5–M7** Quant / encoders / multi-chip | Not started | Device memory stats blocked in PJRT |

Details and targets: [docs/VISION.md](docs/VISION.md). Latest measurements: [docs/PERF.md](docs/PERF.md),
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Good first areas

1. **M4 graph replay:** `scripts/m4_replay_probe.py`, `scripts/trace_cost.py` — see [docs/M4.md](docs/M4.md).
3. Reproduce the matrix on other Tenstorrent cards / hosts.
4. Device-memory introspection for the PJRT plugin (unblocks M5 memory claims).
5. Sampling on the fixed-shape generate path.

## License

By contributing you agree that your contributions are licensed under Apache-2.0, the same as the
project. TensCUDA is an unofficial community project, not affiliated with or endorsed by Tenstorrent Inc.
