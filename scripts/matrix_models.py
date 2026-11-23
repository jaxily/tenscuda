"""Canonical HF model ids for matrix / bench / compile-scale sweeps.

Keep ungated mirrors (`unsloth/*`) where Meta/Google gate the original checkpoint.
Bigger than ~9B bf16 usually needs M5 quantization or multi-chip (VISION M5/M7).
"""

# Current COMPATIBILITY.md ladder (135M → 8B).
MATRIX_12 = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "unsloth/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "unsloth/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
]

# New 7B–9B class to probe whether warm tok/s rises with size under compile unlock.
# Re-includes Llama-3.1-8B so the scale plot has a known anchor.
MATRIX_BIG = [
    "Qwen/Qwen2.5-7B-Instruct",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "unsloth/gemma-2-9b-it",
]

# Apples-to-apples compile tok/s ladder (small → big). Prefer known-good archs first.
COMPILE_SCALE = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "unsloth/gemma-2-9b-it",
]

# Keep-testing extras (after MATRIX_BIG). 14B may OOM bf16 on one P150 — still record.
MATRIX_MORE = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen2.5-14B-Instruct",
]

# Short keys → HF ids for ``python -m tenscuda bench --list``.
BENCH_MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "gemma-2-2b": "unsloth/gemma-2-2b-it",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3",
    "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "gemma-2-9b": "unsloth/gemma-2-9b-it",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}
