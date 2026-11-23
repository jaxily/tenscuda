Notes:

- **CPU fallbacks: aten::_local_scalar_dense** comes from the teacher-forced parity forward
  (`use_cache=False`, the HF mask path reads a scalar). The fixed-shape generate loop that
  produces the tok/s numbers runs with no CPU fallbacks; `python -m tenscuda bench` reports none.
- **Near-ties:** Qwen2.5-0.5B diverges from CPU fp32 at step 16, where CPU's top-2 logits differ by
  0.21. Teacher-forced, TT picks the CPU token at all 32 positions. Qwen2.5-1.5B diverges at step 28,
  where CPU bf16 has an exact tie (margin 0.0) and TT has a 0.25 margin. That's a bf16 coin flip, not a bug.
- **Host RAM:** "anon" is peak RssAnon (sampled every 10 ms), the memory that has to fit in host RAM.
  VmHWM also counts the mmapped safetensors pages. torch-xla defers the host-to-device copy until
  the first execution, so the checkpoint is paged in then as reclaimable page cache.
- **Llama-3.2-1B / 3.2-3B / 3.1-8B** use ungated `unsloth/*` mirrors (same weights) where the
  Meta checkpoints are gated on this host.
- **Gemma-2-2B / Gemma-2-9B** use `unsloth/gemma-2-*-it` (`google/gemma-2-*-it` is gated here). The
  loader rebuilds `Gemma2TextScaledWordEmbedding` buffers with `embed_scale = sqrt(hidden_size)`.
- **Mistral-7B** uses `unsloth/mistral-7b-instruct-v0.3`. **Qwen2.5-7B** is ungated on `Qwen/`.
- **Phi-3.5-mini** greedy stream diverges at step 6 (CPU top-2 margin 0.25); teacher-forced is 31/32.
- **Llama-3.1-8B** CPU reference uses mmap'd bf16 on the 8 GB host (~0.53 GiB peak anon, slow).
  TT streaming load peak **anon** was 0.52 GiB; greedy and teacher-forced **32/32** vs CPU bf16.
- **7B–9B scale:** add rows with `scripts/run_tens_matrix_big.sh` or
  `scripts/run_tens_compile_scale.sh` (`--compile-bench`). 14B+ needs M5 bfp quant on one P150.
- Device memory isn't shown. `xm.get_memory_info` is UNIMPLEMENTED in pjrt-plugin-tt
  1.5.0.dev20260831, and no other in-process route works yet (see docs/PERF.md).
