# FAST: Tenstorrent over CUDA (showcase)

**Purpose:** show why a Blackhole P150 matters for **throughput** — using the **native**
tt-metal / vLLM-TT stack, not TensCUDA’s HF compat path.

TensCUDA (~11 tok/s on Llama-8B) is for “my `.cuda()` script runs.” **This page is the speed story.**

## 30-second talk track

1. Same Llama-3.1-8B on one P150: CPU decode is ~3 tok/s; **native batch-32 hits ~1000 tok/s aggregate** — **~350× CPU**.
2. That is **vLLM-class concurrent serving** (32 sequences), not a single Python loop.
3. Single-user batch-1 on metal is **~40 tok/s**, in the same ballpark as **RTX 3090 FP16** (~45–62 tok/s) for this model class — not a universal “beat every CUDA GPU” claim.
4. Arbitrary Hugging Face models still need **TensCUDA** or porting; **supported Llama paths use vLLM-TT** for volume.

## One command on the card (`tens`)

Prereq: [TT_METAL_FASTPATH.md](TT_METAL_FASTPATH.md) (built `~/tt-metal`, `~/vllm-tt-plugin`). **One job on the card.**

```bash
cd ~/tenscuda
export HF_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
bash scripts/run_tt_speed_showcase.sh
# Optional longer path: bash scripts/run_tt_speed_showcase.sh --serve
```

Output: `/tmp/tt_metal_b32_speed.log`, `/tmp/vllm_offline_b32_speed.log`, and  
`results/showcase_<timestamp>Z.json`.

Re-parse existing logs only:

```bash
bash scripts/run_tt_speed_showcase.sh --quick
```

## Headline numbers (measured P150, Nov 23, 2026)

CPU reference (matrix bf16 greedy): **~3.05 tok/s**.

| Scenario | Metric | Rate | vs CPU |
|---|---|---:|---:|
| **tt-metal batch-32** | aggregate throughput | **~1079 tok/s** | **~354×** |
| **vLLM-TT offline** (32 concurrent) | aggregate / peak gen | **~909 / ~927 tok/s** | **~298–304×** |
| **tt-metal batch-1** | tok/s/user | **~40.5** | **~13×** |
| TensCUDA compile (compat) | batch-1 | ~11–13 | ~4× |

Artifacts: [SPEED.md](SPEED.md), `results/tt_metal_b32_speed_*.json`, `results/vllm_offline_b32_speed_*.json`, `results/showcase_*.json`.

## Fair comparison vs CUDA

| Question | Tenstorrent (P150) | CUDA (typical) |
|---|---|---|
| **Many users at once (32 seq)** | **~900–1079 tok/s** measured | Run same harness: `scripts/nvidia_vllm_bench_ref.sh` on a CUDA box → `results/nvidia_ref_*.json` |
| **One user, batch-1** | **~40 tok/s** metal | **~45–62 tok/s** 3090 FP16 (estimate until measured) |
| **Any HF `.cuda()` script** | TensCUDA (~4× CPU on 8B) | CUDA wins today without porting |

Merge TT + NVIDIA JSON for a slide:

```bash
python scripts/compare_tt_cuda_showcase.py results/showcase_LATEST.json \
  --nvidia results/nvidia_ref_LATEST.json
# → results/showcase_tt_vs_cuda.json + markdown snippet
```

If you have **no NVIDIA GPU**, keep aggregate CUDA as **TBD** in the deck and lead with **350× CPU** on hardware you own.

## What not to claim

- TensCUDA matrix tok/s as “competitive with CUDA serving.”
- **100× CPU** on batch-1 without batching (~300 tok/s is above metal batch-1 today).
- Q4 CUDA vs FP16 metal batch-1 without measuring both.

## Related

| Doc | Role |
|---|---|
| [SPEED.md](SPEED.md) | Canonical numbers + reproduce cmds |
| [TT_METAL_FASTPATH.md](TT_METAL_FASTPATH.md) | Build metal + vLLM env |
| [VISION.md](VISION.md) | TensCUDA compat milestones (M1–M5) |