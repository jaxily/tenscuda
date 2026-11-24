"""Fast (vLLM-TT / metal) vs compat (TT-XLA) routing for approved models."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger("tenscuda.fast_router")

Route = Literal["fast", "compat"]

# HuggingFace ids known good on vLLM-TT offline b32 on P150 (expand from plugin README).
FAST_MODEL_IDS: frozenset[str] = frozenset(
    {
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B",
    }
)

FAST_MODEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^unsloth/Llama-3\.[12]-", re.I),
    re.compile(r"^meta-llama/Llama-3\.[12]-", re.I),
)


def fast_mode() -> str:
    """``off`` | ``advisory`` (default when set) | ``enforce``."""
    v = os.environ.get("TENSCUDA_FAST", "off").strip().lower()
    if v in {"", "0", "false", "no", "off"}:
        return "off"
    if v in {"enforce", "require", "vllm"}:
        return "enforce"
    return "advisory"


def model_id_from_config(config) -> str | None:
    if config is None:
        return None
    for key in ("_name_or_path", "name_or_path"):
        val = getattr(config, key, None)
        if val:
            return str(val)
    return None


def is_fast_eligible(model_id: str | None) -> bool:
    if not model_id:
        return False
    mid = model_id.strip()
    if mid in FAST_MODEL_IDS:
        return True
    return any(p.search(mid) for p in FAST_MODEL_PATTERNS)


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    model_id: str | None
    reason: str

    @property
    def use_fast(self) -> bool:
        return self.route == "fast"


def decide_route(
    *,
    model_id: str | None,
    batch_size: int = 1,
    greedy: bool = True,
    fast_mode_value: str | None = None,
) -> RouteDecision:
    mode = fast_mode_value if fast_mode_value is not None else fast_mode()
    eligible = is_fast_eligible(model_id)

    if not eligible:
        return RouteDecision("compat", model_id, "model not on fast allowlist")
    if mode == "off":
        return RouteDecision("compat", model_id, "TENSCUDA_FAST off")
    if batch_size != 1:
        return RouteDecision(
            "fast",
            model_id,
            "aggregate serving: use vLLM/metal batching, not TT-XLA batch probe",
        )
    if not greedy:
        return RouteDecision("compat", model_id, "sampling requires compat/HF path today")
    if mode in {"advisory", "enforce"}:
        return RouteDecision("fast", model_id, f"TENSCUDA_FAST={mode}")
    return RouteDecision("compat", model_id, "default compat")


def maybe_block_compat_generate(decision: RouteDecision) -> None:
    if decision.route != "fast" or fast_mode() != "enforce":
        if decision.route == "fast" and fast_mode() == "advisory":
            log.warning(
                "tenscuda: model %s is fast-path eligible (~900+ tok/s via vLLM-TT). "
                "This call uses TT-XLA compat (~11 tok/s on 8B). "
                "Use vLLM (see docs/CUDA_TO_METAL.md) or set TENSCUDA_FAST=0 to silence.",
                decision.model_id,
            )
        return
    raise RuntimeError(
        f"TENSCUDA_FAST=enforce: refuse TT-XLA generate for fast-eligible model {decision.model_id!r}. "
        "Run vLLM-TT offline/serve (metal venv) — docs/CUDA_TO_METAL.md."
    )


def vllm_env_hint() -> dict[str, str]:
    return {
        "venv": os.environ.get("VLLM_METAL_VENV", "~/tt-metal/python_env"),
        "plugin": os.environ.get("VLLM_TT_PLUGIN_DIR", "~/vllm-tt-plugin"),
        "showcase": "bash scripts/run_tt_speed_showcase.sh --skip-metal",
    }