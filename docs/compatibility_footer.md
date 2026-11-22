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
- **Llama-3.2-1B** uses the ungated `unsloth/Llama-3.2-1B-Instruct` mirror (same weights);
  `meta-llama/Llama-3.2-1B-Instruct` is gated for this host's token.
- Device memory isn't shown. `xm.get_memory_info` is UNIMPLEMENTED in pjrt-plugin-tt
  1.5.0.dev20260831, and no other in-process route works yet (see docs/PERF.md).
