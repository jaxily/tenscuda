# Native TT-metal fast path (for speed)

TensCUDA uses **TT-XLA**. For peak tok/s on supported models, run **tt-metal / tt-transformers**
on the same P150 — a separate checkout and venv. Do not mix that venv with TensCUDA’s
`.venv` (pjrt-plugin-tt).

**Measured ≥100× CPU summary:** [SPEED.md](SPEED.md). Research context: [SPEED_RESEARCH.md](SPEED_RESEARCH.md).

## On `tens` (as of Nov 23, 2026)

| Item | Status |
|---|---|
| KMD | `tenstorrent-dkms` 2.11.0, `/dev/tenstorrent/0` present |
| OS | Ubuntu 24.04 (Noble) — source build supported (`g++-14`) |
| Docker | not installed |
| System `git` | missing — user-local git at `~/.local/bin/git` (from apt `.deb` extract) |
| `tt-smi` | not on PATH (optional; install via tt-installer if desired) |
| Cached weights | `unsloth/Meta-Llama-3.1-8B-Instruct` already in HF hub cache |
| Checkout | `~/tt-metal` (ttt_compat demos) |
| Build / venv | **Done** (after `install_dependencies` + `build_metal` + `create_venv`) |
| Measured speed | **See below** — batch-1 and batch-32 on one P150 |

Upstream rename: demos live under **`models/ttt_compat/`** (old `models/tt_transformers`
is a deprecated import alias). New feature work is in the separate
[tt-transformers](https://github.com/tenstorrent/tt-transformers) repo; for P150 Llama-8B
speed, `ttt_compat` demos are still the right first target.

## Measured on `tens` (Nov 23, 2026) — Llama-3.1-8B

Host: one Blackhole P150, `HF_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct`,
`models/ttt_compat/demo/simple_text_demo.py`, performance mode.

| Config | Per-user | Aggregate | vs CPU (~3 tok/s) | Result |
|---|---:|---:|---:|---|
| **batch-1** (`…-performance-batch-1`) | **~40.5 tok/s/u** | **~40.5 tok/s** | **~13×** | PASSED |
| **batch-32** (`…-performance-batch-32`) | **~33.7 tok/s/u** | **~1079 tok/s** | **~350×** | PASSED |

**Reconfirm (Nov 23 13:09, hot JIT):** Average **29.66 ms @ 33.72 tok/s/user (1079.0 tok/s)**;
TTFT 14.31 ms; PASSED in 33.8 s. Log `/tmp/tt_metal_b32_speed.log`; JSON
`results/tt_metal_b32_speed_2026-11-23T1309.json`.

**vs TensCUDA on the same card (Llama-8B compile ~11–13 tok/s ≈ 4× CPU):** metal batch-1 is
~**3×** faster than TensCUDA; metal batch-32 aggregate is ~**80–100×** TensCUDA and
**≥100× CPU** (the speed claim).

**vs published TT P150 table (~33 t/s/u, ~1059 t/s @ b32):** we match or beat it.
**vs RTX 3090 FP16 (~45–62 tok/s batch-1):** single-user is in the same ballpark; Q4 on
3090 still pulls ahead on batch-1.

Run **only** the node id (do not also pass the bare `.py` file — that re-collects other
tests and previously caused a runaway batch-1 decode):

```bash
# Prefer -k (parametrize brackets break bare ::node ids under some launchers)
pytest models/ttt_compat/demo/simple_text_demo.py -k "performance and batch-32 and not log-probs" -s
```

## Setup (source — recommended for demos + kernel work)

```bash
# PATH for user-local git if system git missing
export PATH="$HOME/.local/bin:$PATH"
export GIT_EXEC_PATH="$HOME/.local/opt/git-deb/usr/lib/git-core"

# 1) Clone (done on tens)
# git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal

# 2) System deps — YOU must run this (sudo password)
sudo ./install_dependencies.sh

# 3) Build (~30–60 min first time)
./build_metal.sh

# 4) Python env (separate from TensCUDA)
./create_venv.sh
source python_env/bin/activate
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME
# single Blackhole; mesh name may be auto-detected — check demo logs if unset

# 5) Model deps — note: this tree has NO models/ttt_compat/requirements.txt
# create_venv.sh already installs tt_metal/python_env/requirements-dev.txt
# If demo imports fail, install missing packages ad hoc (transformers, etc.)
```

Pre-pytest checks:

```bash
# Card free?
ps aux | grep -iE 'matrix|pjrt|tenscuda|pytest' | grep -v grep

ls models/ttt_compat/demo/simple_text_demo.py   # must exist (it does on 4147e37)
python3 -m ttnn.examples.usage.run_op_on_device
```

## Speed demos (Llama 3.1 8B)

```bash
cd ~/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=~/tt-metal PYTHONPATH=$TT_METAL_HOME
export HF_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct

# Single-user — expect ~20–35 t/s/u; watch for === Performance metrics ===
pytest models/ttt_compat/demo/simple_text_demo.py -k "performance and batch-1"

# Multi-user aggregate (~1000 tok/s class @ batch 32)
pytest models/ttt_compat/demo/simple_text_demo.py -k "performance and batch-32"
```

Log measured tok/s next to TensCUDA’s matrix row (~3–5 tok/s on 8B) in `results/` and
[PERF.md](PERF.md).

## Rules of engagement

1. **One job on the card** — do not run TensCUDA `matrix.py` and tt-metal demos together
   (`CHIP_IN_USE`).
2. **Separate venvs** — TensCUDA `.venv` stays for TT-XLA; `tt-metal/python_env` for metal.
3. **Kernel edits** live in `~/tt-metal` (or upstream PRs). TensCUDA still only *routes* to
   ttnn via TT-XLA when we land M4 E4 — it does not own metal C++.

## vLLM-TT (`tens`, Nov 23)

Installed in **`~/tt-metal/python_env`** (not TensCUDA `.venv`):

```bash
git clone https://github.com/tenstorrent/vllm-tt-plugin.git ~/vllm-tt-plugin
cd ~/vllm-tt-plugin && source ~/tt-metal/python_env/bin/activate && bash docs/install-vllm-tt.sh
```

Or `scripts/setup_vllm_tt_tens.sh` from the TensCUDA repo (same steps).

**Batch concurrent TPS** (target: same class as pytest batch-32 ~1079 tok/s):

1. One job on the card — stop TensCUDA matrix before serving.
2. Start server, e.g. `MESH_DEVICE=N150 python examples/server_example_tt.py` with
   `--model unsloth/Meta-Llama-3.1-8B-Instruct` and `--max_seqs_in_batch 32` (see plugin README).
3. `vllm bench serve --model … --num-prompts 32 --random-input-len 128 --random-output-len 128 …`

**Measured (Nov 23 13:12, 256 prompts, 32 concurrent, 128 max new tokens):**

| Metric | Value | vs CPU (~3 tok/s) |
|---|---:|---:|
| Aggregate output tok/s (vLLM tqdm) | **~909** | **~298x** |
| Engine peak generation tok/s | **~927** | **~304x** |
| pytest `ttt_compat` batch-32 (reference) | **~1079** | **~354x** |

Cmd (requires `HF_MODEL` + `trace_region_size` ≥ ~61MB on P150):

```bash
export HF_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct MESH_DEVICE=N150
python examples/offline_inference_tt.py --model "$HF_MODEL" \
  --prompts_json examples/prompts.json --num_repeat_prompts 8 \
  --max_seqs_in_batch 32 --greedy_sampling --max_tokens 128 --max_model_len 8192 \
  --additional-config '{"tt": {"trace_region_size": 67108864}}'
```

Artifacts: `results/vllm_offline_b32_speed_2026-11-23T1312.json`, log `/tmp/vllm_offline_b32_speed.log` on `tens`.

## Optional later

- Further vLLM models / multi-chip meshes per plugin README.
- [tt-transformers](https://github.com/tenstorrent/tt-transformers) for new model bring-up
  (v1 code is frozen as `ttt_compat`).
- `tt-smi` via [tt-installer](https://github.com/tenstorrent/tt-installer) for telemetry
  alongside TensCUDA Monitor.
