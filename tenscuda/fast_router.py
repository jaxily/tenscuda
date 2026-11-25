"""Fast (vLLM-TT / metal) vs compat (TT-XLA) routing for approved models."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger("tenscuda.fast_router")

Route = Literal["fast", "compat", "delegate"]

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


def _normalize_fast_mode(raw: str) -> str:
    v = raw.strip().lower()
    if v in {"", "0", "false", "no", "off"}:
        return "off"
    if v in {"enforce", "require"}:
        return "enforce"
    if v in {"delegate", "vllm", "native"}:
        return "delegate"
    if v in {"aggregate", "b32", "serving"}:
        return "aggregate"
    if v == "auto":
        return "auto"
    if v == "advisory":
        return "advisory"
    return "advisory"


def fast_mode() -> str:
    """``off`` | ``advisory`` | ``enforce`` | ``delegate`` | ``aggregate`` | ``auto``."""
    return _normalize_fast_mode(os.environ.get("TENSCUDA_FAST", "off"))


def _vllm_delegate_modes() -> frozenset[str]:
    return frozenset({"delegate", "auto", "enforce", "aggregate"})


def use_aggregate_client() -> bool:
    """True when decode should saturate vLLM with concurrent HTTP (target ~900+ tok/s)."""
    return fast_mode() == "aggregate"


def default_aggregate_concurrency() -> int:
    return int(os.environ.get("TENSCUDA_AGGREGATE_CONCURRENCY", "32"))


def model_id_from_config(config) -> str | None:
    if config is None:
        return None
    for key in ("_name_or_path", "name_or_path"):
        val = getattr(config, key, None)
        if val:
            return str(val)
    return None


def skip_xla_weights_load(model_id: str | None) -> bool:
    """True when ``load_causal_lm`` should not stream weights to the card (vLLM owns weights)."""
    if not model_id:
        return False
    mid = model_id.strip()
    if not is_fast_eligible(mid):
        return False
    return fast_mode() in _vllm_delegate_modes()


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
        return self.route in {"fast", "delegate"}


def decide_route(
    *,
    model_id: str | None,
    batch_size: int = 1,
    greedy: bool = True,
    fast_mode_value: str | None = None,
) -> RouteDecision:
    mode = _normalize_fast_mode(fast_mode_value) if fast_mode_value is not None else fast_mode()
    eligible = is_fast_eligible(model_id)

    if not eligible:
        return RouteDecision("compat", model_id, "model not on fast allowlist")
    if mode == "off":
        return RouteDecision("compat", model_id, "TENSCUDA_FAST off")
    if batch_size != 1:
        if mode in _vllm_delegate_modes():
            return RouteDecision(
                "delegate",
                model_id,
                f"TENSCUDA_FAST={mode} batch={batch_size} -> vLLM concurrent",
            )
        return RouteDecision(
            "fast",
            model_id,
            "aggregate serving: set TENSCUDA_FAST=aggregate and run vLLM serve",
        )
    if not greedy:
        return RouteDecision("compat", model_id, "sampling requires compat/HF path today")
    if mode in _vllm_delegate_modes():
        label = "aggregate concurrent client" if mode == "aggregate" else "vLLM-TT"
        return RouteDecision("delegate", model_id, f"TENSCUDA_FAST={mode} -> {label}")
    if mode == "advisory":
        return RouteDecision("fast", model_id, "TENSCUDA_FAST=advisory")
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
        "Set TENSCUDA_FAST=delegate and run vLLM-TT serve — docs/CUDA_TO_METAL.md."
    )


def vllm_env_hint() -> dict[str, str]:
    return {
        "venv": os.environ.get("VLLM_METAL_VENV", "~/tt-metal/python_env"),
        "plugin": os.environ.get("VLLM_TT_PLUGIN_DIR", "~/vllm-tt-plugin"),
        "showcase": "bash scripts/run_tt_speed_showcase.sh --skip-metal",
    }