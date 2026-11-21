# TensCUDA performance & learnings log

Hardware: one Tenstorrent Blackhole P150a on PCIe 3.0 x16. Host: 4-core Intel i7-6700, 8 GB RAM.
Stack: tt-metal v0.82, KMD 2.11.0, pjrt-plugin-tt 1.5.0.dev20260831, torch-xla 2.9.0, torch 2.11.0, transformers 5.17.0.
The prompt is a chat template plus "Explain in two sentences why the sky is blue." Decoding is greedy, 32 new tokens, batch 1.
"Warm" means the runs after the first one in the same process. Each experiment was repeated at least 3x. Raw JSON is in `results/`.

## 1. Getting a working path

| Finding | Detail |
|---|---|
| Route | TT-XLA: `pjrt-plugin-tt` + TT's torch-xla build, in an isolated venv. No system packages changed. |
| Install quirk | The plugin pins torch, torch-xla and tt-lang via direct URLs, so `uv` needs them as explicit requirements. |
| SFPI mismatch | The system SFPI 7.93 rejected the plugin's kernels (`-Werror` on a deprecated macro). Installing SFPI 7.69.0 inside the venv's `pjrt_plugin_tt/tt-metal/runtime/sfpi` fixed it, and the system wasn't touched. |
| Harmless warnings | `Setting power state failed ... Remote I/O error`, unknown motherboard, and an ethernet firmware 1.6.0 note about single-erisc fallback. |

## 2. Correctness bugs found and fixed

| Bug | Symptom | Fix |
|---|---|---|
| Python-wrapping `torch.zeros/empty/tensor/...` | CPU results changed silently (the first SmolLM token went 504 -> 1) because torch's `with torch.device(...)` context keys on the original builtins | Use one `TorchFunctionMode` that rewrites `device="cuda"` kwargs instead |
| `torch.cuda.manual_seed_all` -> `torch.manual_seed` | Infinite recursion, since torch.manual_seed calls cuda.manual_seed_all | Seed the XLA RNG directly |
| Dynamo + `cuda.is_available()==True` | `get_rng_state` IndexError | Stub the CUDA RNG state functions |
| FX codegen + `torch.device` proxy | `NameError: device` in generated code | Restore the real `torch.device` around compiled calls |
| Fused TT softmax | 2.26e-2 rel L2 error even in fp32 (SDPA 2.44e-2) | `exp(x-max) * reciprocal(sum)`: 2.76e-4, SDPA 1.70e-3 |
| Lazy graph never cut in the decode loop | Step time grew linearly (0.39 -> 1.37 s) and 8 compiles took 450 s | `torch_xla.sync()` after each step |

Important: `exp(x-max) / sum` is pattern-matched back into the same inaccurate fused op (2.26e-2).
Multiplying by the reciprocal avoids the match. Plain division, reciprocal, rsqrt, exp and silu are all accurate on their own.

## 3. Op-level parity (TT vs CPU fp32; `scripts/op_parity.py`, HiFi4 + fp32 accumulation)

| Op | fp32 rel L2 | bf16 rel L2 |
|---|---|---|
| matmul 256x384x192 | 1.17e-3 | 1.70e-3 |
| linear 4x256x512 -> 1024 | 1.10e-3 | 1.69e-3 |
| attention scores q@k^T | 1.17e-3 | 1.69e-3 |
| layer_norm 512 | 1.18e-3 | 3.29e-3 |
| rms_norm 512 | 1.48e-3 | 1.91e-3 |
| softmax (fused, before fix) | 2.26e-2 | 2.30e-2 |
| softmax (TensCUDA) | 2.76e-4 | |
| SDPA causal (fused, before fix) | 2.46e-2 | 2.41e-2 |
| SDPA causal (TensCUDA) | 1.70e-3 | |
| silu / exp | 6e-8 / 3e-8 | 1.8e-3 / 2.0e-3 |
| gelu-tanh | 2.0e-4 | 1.7e-3 |
| embedding gather | 1.65e-3 | 0 |
| cumsum | 3.4e-7 | 1.7e-3 |
| argmax | 0 | 0 |
| H2D/D2H roundtrip | bit-exact (fp32, bf16, int32, int64) | |

fp32 matmuls are TF32-class (about 1e-3), not IEEE fp32. Math fidelity does take effect: LoFi raises matmul error to 3.1e-2.

## 4. Token parity

| Model | TT dtype | vs CPU fp32 | vs CPU bf16 | Notes |
|---|---|---|---|---|
| SmolLM2-135M | fp32 | 32/32 | | |
| SmolLM2-135M | bf16, fused softmax | 2/32 | 2/32 | The flip was at a near-tie: fp32 margin 0.12 logit, which is below one bf16 step (0.125) at 29.5 |
| SmolLM2-135M | bf16, TensCUDA softmax | **32/32** | 30/32 | CPU bf16 is the outlier: at token 30 CPU fp32 says " like" (26.76 vs 26.65); CPU bf16 KV-cache decode picks " in" |
| TinyLlama-1.1B | bf16 | **32/32** | **32/32** | Streaming loader |

Per-layer hidden-state check (`scripts/layer_diff.py`, SmolLM2 bf16, fused softmax, at the divergent step):
TT error vs fp32 was 1.0-2.3e-2 per layer, against CPU bf16's own 0.06-3.3e-2, with no single
layer jumping out. In a full (no-cache) forward pass TT already picked the fp32 token. The
flip was rounding plus the fused-softmax error, not a broken op.

## 5. Speed experiments (SmolLM2-135M bf16, 33 tokens, warm decode tok/s, 3 repeats)

| Idea | Warm decode tok/s | First run | Parity | Kept? |
|---|---|---|---|---|
| Stock HF generate (dynamic cache) | not usable: one new compile per token, over 15 min for 32 tokens | | | no |
| Static KV cache + lazy tensors (baseline) | 3.54 / 3.56 / 3.59 | 50 s (5 graphs, 39 s compile) | 32/32 | **yes** |
| Device-resident tokens and positions, no per-token readback (K=1) | 3.60 / 3.56 / 3.51 | 53 s | 32/32 | yes (no gain) |
| K=4 decode steps per graph | 3.94 / 3.96 / 3.93 | 106 s (96 s compile) | **wrong tokens** | no |
| K=8 decode steps per graph | 3.85 / 3.84 / 3.80 | 192 s | **wrong tokens** | no |
| `torch.compile(backend="tt")` | 2.62 / 2.62 / 2.61 | 199 s | **wrong from token 1** | no |
| `torch.compile(backend="openxla")` | 2.62 / 2.60 / 2.61 | 130 s | **wrong** | no |

Learnings:
- **Host-bound.** A py-spy profile of steady decode puts about 94% of the time in Python/torch-xla
  tracing of the HF model (tt_torch's TorchFunctionOverride, TensCUDA's mode, einsum, rotary,
  HF wrappers). Only about 6% is in `torch_xla.sync` (device execute + wait). Reading the token
  back each step isn't the bottleneck, so device residency alone gains nothing.
- Multi-step graphs and dynamo both corrupt HF `StaticCache` in-place updates (the likely
  cause is aliasing or mutation handling across steps). Fixing this, or using a functional
  KV cache, is the precondition for graph replay, the main remaining speed lever.
- Compile cost: the first run compiles 4-5 graphs (prefill, two decode variants, small helpers).
  That takes about 23-39 s, plus the kernel JIT cache in `~/.cache/tt-metal-cache` (about 1.8 GB
  after these runs).

## 6. Loading

| Model | Loader | Load time | Peak RSS after load | Peak RSS end of run |
|---|---|---|---|---|
| SmolLM2-135M | streaming | 0.7 s | 0.70 GiB | 1.10 GiB |
| SmolLM2-135M | `from_pretrained().cuda()` | 0.8 s | 0.70 GiB | 1.09 GiB |
| TinyLlama-1.1B (2.05 GiB bf16) | streaming | 1.0 s | 0.68 GiB | 2.5 GiB |
| TinyLlama-1.1B, CPU fp32 reference | (CPU) | | | 4.16 GiB |

The streaming loader builds the model on `meta`, uploads one safetensors tensor at a time and
rebuilds non-persistent buffers (rotary `inv_freq`) on the CPU. Most of the end-of-run RSS growth
comes from graph compilation (TT-MLIR), not from weights.

## 7. Quantization (bfp8 / bfp4): UNVERIFIED

- In lazy mode, `torch.ops.tt.weight_dtype_override` on a weight alone has **no effect**. A
  single 2048x2048 linear gives rel err 2.88e-3 for bf16, "bfp8" and "bfp4" alike, and TinyLlama
  logits are identical across all three.
- Marking the weight as a `parameter` argument (`tt.mark_argument_attributes`) as well makes
  the conversion happen. We saw about 7.9e-3 (bfp8-like) and 1.11e-1 (bfp4-like) errors.
- The results then depend on which variant compiled first. The graph cache seems to ignore the
  dtype attribute, so a later variant with the same structure reuses the earlier executable.
  Each variant must be measured in a fresh process, which hasn't been done yet.
- Device memory can't be measured yet: `xm.get_memory_info` returns UNIMPLEMENTED on this plugin.
