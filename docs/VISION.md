# TensCUDA: vision, goals and roadmap

> TensCUDA is an unofficial community project and is not affiliated with or endorsed by Tenstorrent Inc.

---

## Community announcement

*This is a community project announcement, not an official Tenstorrent press release.*

### TensCUDA: unmodified PyTorch `.cuda()` scripts now run on Tenstorrent Blackhole

**An open-source adapter that maps PyTorch's CUDA device API onto Tenstorrent cards through TT-XLA. Llama-family models run from stock Hugging Face scripts, with token output identical to the CPU.**

Most AI code on the internet is written for CUDA. It calls `.cuda()`, checks
`torch.cuda.is_available()` and creates tensors with `device="cuda"`. TensCUDA is a small
Apache-2.0 Python package that redirects that API to a Tenstorrent Blackhole card. You
either add one line, `import tenscuda`, or launch an untouched script with
`python -m tenscuda script.py`. Compilation is handled by Tenstorrent's open-source TT-XLA
stack (torch-xla + `pjrt-plugin-tt`).

On a single Blackhole P150a in a 7.5 GB-RAM host, unmodified Hugging Face greedy-generation
scripts for **SmolLM2-135M** and **TinyLlama-1.1B** produce **32 of 32 tokens identical to
the CPU reference** (fp32 for SmolLM2; fp32 and bf16 for TinyLlama). TinyLlama decodes at
**about 4.2 tokens/s warm**. A streaming loader moves weights from safetensors straight to
the card's memory, with a host peak of **0.68 GiB RSS after load**.

Getting there surfaced real, reproducible findings in the stack. Wrapping torch factory
functions silently corrupted Hugging Face model initialization. The fused softmax on this
stack has about 2.3e-2 relative error, and the same holds in fp32. bfp8/bfp4 weight
annotations do nothing unless the weight is also marked as a parameter argument. Each one has
a fix or workaround in the repo, backed by a test or a measurement.

**What you can do today**
- Run a stock HF causal-LM script on a P150 with `python -m tenscuda script.py`.
- Load models straight to the card with `tenscuda.loader.load_causal_lm`, without staging them in host RAM.
- Use accurate softmax/SDPA on TT by default, and op-level parity checks (`scripts/op_parity.py`).
- Try bfp8 weights (`tenscuda.quant.quantize_weights`). On TinyLlama they keep 32/32 token parity; this feature is experimental.
- Add any model to the compatibility matrix with one command (`scripts/matrix.py MODEL`).

---

## Purpose and problem

The barrier to adopting new AI accelerators is usually software, not silicon. Tutorials, model
repos, research code and internal tools assume CUDA. Porting each script by hand to a new
backend is slow and fragile, so capable hardware sits idle while users wait for official
support for each model.

TensCUDA attacks the most common case: PyTorch code that uses CUDA only as a *device*
(moving tensors and modules, checking availability, synchronizing), not as a kernel language.
For that code the device API can be remapped, and an existing compiler (TT-XLA) can do the rest.

## Business case: why we focused here

**PyTorch is where a large share of AI developers and models live.** The PyTorch Foundation reports:

| Reach indicator | Reported figure | Source |
|---|---|---|
| Developers and organizations | Over **150,000 developers** and **18,000 organizations** | [PyTorch: a Foundation for Open Source AI/ML](https://pytorch.org/event/webinar-pytorch-a-foundation-for-open-source-ai-ml/) |
| Model training adoption | **63%** adoption rate, from the Linux Foundation's *Shaping the Future of Generative AI* report | [2024 Year in Review](https://pytorch.org/blog/2024-year-in-review/) |
| AI research implementations | Over **70%** use PyTorch | [2024 Year in Review](https://pytorch.org/blog/2024-year-in-review/) |
| GitHub repositories | Over **600,000** repos | [One Year of PyTorch Foundation](https://pytorch.org/blog/one-year-pytorch/) |

These are published historical figures, not a 2026 market-size estimate. They show the scale of the
potential developer onramp, not adoption that TensCUDA has achieved.

A great deal of PyTorch code, including Hugging Face GPU examples, assumes CUDA through `.cuda()`,
`device="cuda"`, CUDA availability checks and tensor factories. We haven't counted the fraction of
all examples that do this, but it's the common path a new accelerator must meet. A user who can run
that code unchanged can try Tenstorrent hardware immediately; a user who has to port it first may
never start.

**Our thesis: porting cost is the first adoption barrier to remove, before optimizing hardware
price/performance.** One device-API adapter can open the door to many users and model families at
once, rather than one manual port at a time. That's potential reach, not a claim that every PyTorch
model already runs. Unsupported ops, CUDA extensions and model-specific compiler gaps remain.

Every model that runs becomes a compatibility-matrix entry, a reproducible test and a demand signal
for native ttnn kernels. It gives Tenstorrent engineers concrete workloads to optimize, and gives
users an honest map of what works. That pairs naturally with Tenstorrent's open-source software and
accessible-hardware positioning: lower the software switching cost as well as the hardware entry cost.

The small-host-RAM story matters too. Weights can stream from safetensors into the P150's 32 GB GDDR6
one tensor at a time instead of requiring equivalent host RAM. TinyLlama's measured 0.68 GiB RSS
after load is the first proof; 8B under 2 GiB *end-to-end* host RSS is a target, not a result.

## Value to Tenstorrent

- **An onramp for CUDA and PyTorch users.** Developers can try a Tenstorrent card with code they
  already have, instead of learning a new API first.
- **Big models on small hosts.** Weights stream straight to the card's 32 GB GDDR6, so the
  host's RAM isn't the limit. The test host has under 8 GB.
- **A real-world compatibility matrix** ([COMPATIBILITY.md](COMPATIBILITY.md)). Every model row
  records parity, speed, compile time and memory. That shows concretely where TT-XLA, tt-mlir
  and ttnn are strong and where gaps remain.
- **A funnel to native kernels.** Profiles show exactly which ops and phases dominate. Those
  are the candidates to route to native ttnn/tt-metal kernels, so users move gradually from
  "it runs" to "it runs fast".
- **Bug reports and fixes upstream.** These are findings so far, each with a minimal
  reproduction in the repo:
  - Python-wrapped torch factories break `torch.device` contexts and corrupt HF initialization.
  - The fused softmax error (2.26e-2) and its pattern-match trigger (`exp/sum`) versus the accurate `exp*reciprocal(sum)`.
  - `weight_dtype_override` is a no-op without the `parameter` argument mark.
  - The in-process graph cache ignores weight-dtype attributes.
  - HF `StaticCache` produces wrong tokens under multi-step graphs and `torch.compile(backend="tt")`.

## Goals and milestones

**These are targets, not results.** Each milestone is done when its numbers are measured and
committed to `docs/PERF.md` and `docs/COMPATIBILITY.md`.

| Milestone | Target (measurable) |
|---|---|
| **M1. Tier-1 compatibility matrix** | SmolLM2-360M, SmolLM2-1.7B, Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B each run from an unmodified script, with parity vs CPU, warm tok/s, first-run time and host RSS recorded |
| **M2. 3-4B class** | Llama-3.2-3B, Qwen2.5-3B, Phi-3.5-mini, Gemma-2-2B in the matrix, with greedy parity vs CPU (or vs a published reference where the CPU can't fit them) |
| **M3. 8B on one card from a small host** | Llama-3.1-8B from an unmodified script on one P150, under 2 GiB peak host RSS, with parity on a short prompt |
| **M4. Native kernels** | Route matmul/linear, SDPA, RMSNorm and rotary to native ttnn kernels where TT-XLA is slower, measuring the speedup and parity at each step; remove per-token host re-tracing (graph replay) |
| **M5. Quantization to fit 14B** | bfp8/bfp4 weights verified for accuracy (done for TinyLlama: bfp8 32/32) **and** for device-memory savings (not yet measurable), then a 14B-class model on one card |
| **M6. Beyond LLMs** | Embedding models, Whisper, CLIP and ViT in the matrix |
| **M7. Multi-chip** | Two linked P150s (64 GB) running one model |

## Non-goals and honest limits

- **Not a CUDA kernel or PTX emulator.** Code that ships CUDA kernels (flash-attn, bitsandbytes,
  xformers, custom extensions, many ComfyUI nodes) won't run.
- **Not faster than a CPU on tiny models today.** Decoding is host-bound. torch-xla re-traces the
  Python model every token (about 0.25 s/step on a 2015-era 4-core host) while the card is busy
  for about 6% of the step. SmolLM2-135M runs at 3.6 tok/s on TT versus 16.7 tok/s on the CPU.
  TinyLlama runs at about 4.2 versus 5.5 tok/s for CPU bf16.
- **Not a replacement for Tenstorrent's optimized model implementations.** Models with native
  tt-metal/vLLM support will stay faster; TensCUDA is for everything else.
- **The fast path is single-card, batch 1 and greedy.** Sampling falls back to stock HF `generate`.
- Device-memory statistics aren't available from the plugin yet.

## How Tenstorrent could help

- **Development hardware**, especially a multi-chip system for M7 and faster hosts to separate
  host-bound effects from device performance.
- **Kernel guidance.** The sanctioned way to call native ttnn ops from a TT-XLA graph
  (custom-call or op override), and a recommended graph-replay or trace-capture path for decode loops.
- **Upstream review** of the findings above, and of the patches as they're ready.
- **Device-memory introspection** through the PJRT plugin (`get_memory_info`).

## How the community can contribute

- Run `python scripts/matrix.py <model>` on your card and open a PR with the new row.
- Report models that fail, with the log from `results/`.
- Pick a roadmap item, especially graph replay (M4) or new architectures (M1, M2, M6).
- Keep claims honest: every number in this repo should come from a committed measurement.

---

TensCUDA is an unofficial community project and is not affiliated with or endorsed by Tenstorrent Inc.
