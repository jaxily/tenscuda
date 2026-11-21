# Contributing to TensCUDA

Thanks for helping make Tenstorrent cards easier to use from PyTorch.

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

## Good first areas

1. Reproduce the model matrix on other Tenstorrent cards / hosts.
2. Add small-model and architecture coverage (see `docs/VISION.md`).
3. Find a correct graph-replay path for decode (HF StaticCache mutation currently fails under dynamo).
4. Device-memory introspection for the PJRT plugin.
5. Sampling on the fixed-shape generate path.

## License

By contributing you agree that your contributions are licensed under Apache-2.0, the same as the
project. TensCUDA is an unofficial community project, not affiliated with or endorsed by Tenstorrent Inc.
