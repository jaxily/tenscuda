import os

import pytest

from tenscuda.fast_router import (
    RouteDecision,
    decide_route,
    fast_mode,
    is_fast_eligible,
    maybe_block_compat_generate,
)


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("unsloth/Meta-Llama-3.1-8B-Instruct", True),
        ("meta-llama/Llama-3.2-1B-Instruct", True),
        ("HuggingFaceTB/SmolLM2-135M-Instruct", False),
        ("unsloth/gemma-2-9b-it", False),
    ],
)
def test_allowlist(model_id, expected):
    assert is_fast_eligible(model_id) is expected


def test_decision_matrix():
    d = decide_route(
        model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
        batch_size=1,
        greedy=True,
        fast_mode_value="advisory",
    )
    assert d.route == "fast"
    d2 = decide_route(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        batch_size=1,
        greedy=True,
        fast_mode_value="advisory",
    )
    assert d2.route == "compat"
    d3 = decide_route(
        model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
        batch_size=1,
        greedy=False,
        fast_mode_value="advisory",
    )
    assert d3.route == "compat"


def test_enforce_blocks():
    d = RouteDecision(
        "fast",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "TENSCUDA_FAST=enforce",
    )
    os.environ["TENSCUDA_FAST"] = "enforce"
    try:
        with pytest.raises(RuntimeError, match="enforce"):
            maybe_block_compat_generate(d)
    finally:
        del os.environ["TENSCUDA_FAST"]


def test_enforce_routes_delegate_not_xla():
    d = decide_route(
        model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
        batch_size=1,
        greedy=True,
        fast_mode_value="enforce",
    )
    assert d.route == "delegate"


def test_fast_mode_values():
    os.environ["TENSCUDA_FAST"] = "advisory"
    try:
        assert fast_mode() == "advisory"
    finally:
        del os.environ["TENSCUDA_FAST"]


def test_fast_mode_delegate_aliases():
    for raw, want in (("delegate", "delegate"), ("vllm", "delegate"), ("native", "delegate")):
        os.environ["TENSCUDA_FAST"] = raw
        try:
            assert fast_mode() == want
        finally:
            del os.environ["TENSCUDA_FAST"]