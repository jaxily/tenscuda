import os

import pytest


def test_skip_xla_weights_when_delegate():
    from tenscuda.fast_router import skip_xla_weights_load

    os.environ["TENSCUDA_FAST"] = "aggregate"
    try:
        assert skip_xla_weights_load("unsloth/Meta-Llama-3.1-8B-Instruct")
        assert not skip_xla_weights_load("HuggingFaceTB/SmolLM2-135M-Instruct")
    finally:
        del os.environ["TENSCUDA_FAST"]


def test_stub_rejects_when_not_delegate_mode():
    from tenscuda.loader import load_causal_lm_delegate_stub

    os.environ["TENSCUDA_FAST"] = "off"
    try:
        with pytest.raises(RuntimeError, match="delegate mode"):
            load_causal_lm_delegate_stub("unsloth/Meta-Llama-3.1-8B-Instruct")
    finally:
        del os.environ["TENSCUDA_FAST"]