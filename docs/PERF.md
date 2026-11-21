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

## 7. Quantization (bfp8 / bfp4): works with a required fix. Accuracy is verified; memory saving is NOT verified.

**The trap (why it first looked like it worked).** In lazy mode, `torch.ops.tt.weight_dtype_override`
alone is a no-op. tt-mlir only converts weights whose graph argument is marked `parameter`.
The early "quantized" TinyLlama runs (rel err 0.03654 for bf16, bfp8 and bfp4 alike, identical
text) were really bf16.

**The second trap.** torch-xla's in-process graph cache ignores the dtype attribute. When bf16,
bfp8 and bfp4 graphs with the same structure run in one process, they all reuse the first
compiled executable. Every variant below ran in its own process.

Single 2048x2048 linear, 32 tokens, fresh process per variant (`scripts/quant_probe.py`):

| Variant | rel L2 vs CPU fp32 |
|---|---|
| bf16 | 2.88e-3 |
| override bfp8 / bfp4 only | 2.88e-3 / 2.88e-3 (no-op) |
| `parameter` mark only | 2.88e-3 |
| override + `parameter` mark, bfp8 | 7.86e-3 |
| override + `parameter` mark, bfp4 | 1.11e-1 |

TinyLlama-1.1B with all 154 decoder Linear weights converted (`tenscuda.quant.quantize_weights`,
streaming loader). Logits are teacher-forced against CPU fp32 over prompt + 32 tokens
(`scripts/quant_logits.py`); greedy tokens are compared with CPU fp32 (`scripts/load_bench.py`).

| Weights | logits rel L2 | argmax agree (generated) | KL (generated) | greedy tokens vs CPU fp32 | warm decode tok/s (3 runs) | first run |
|---|---|---|---|---|---|---|
| bf16 | 2.32e-2 | 100% | 2.3e-4 | 32/32 | 4.51 / 3.21* / 4.44 | 57 s |
| bfp8 | 2.74e-2 | 100% | 6.5e-4 | **32/32** | 3.90 / 4.03 / 4.02 | 67 s |
| bfp4 | 2.38e-1 | 87.9% | 1.1e-1 | 6/32 (diverges, still fluent) | 4.09 / 4.05 / 4.03 | 63 s |

(*) That run overlapped a model download on the same host.

- bfp8 is close to lossless at this size, and bfp4 is clearly lossy.
- Speed doesn't change, because decoding is host-bound (section 5).
- Host RSS doesn't change either: weights still upload as bf16 (about 0.69 GiB after load in all modes).
- **Device memory saving is unmeasured.** `xm.get_memory_info` returns UNIMPLEMENTED, and the
  conversion is a typecast inside the compiled graph, so the bf16 originals may stay resident.
  Until device DRAM can be measured, don't count on bfp to fit bigger models.
