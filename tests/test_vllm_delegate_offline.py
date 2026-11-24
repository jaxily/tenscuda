import json
from unittest.mock import patch

import torch

from tenscuda.fast_router import RouteDecision, decide_route
from tenscuda.vllm_delegate import VllmDelegateError, delegate_greedy_generate, probe_vllm, vllm_base_url


class _MiniModel:
    config = type("C", (), {"name_or_path": "unsloth/Meta-Llama-3.1-8B-Instruct"})()

    def parameters(self):
        yield torch.zeros(1)


def test_decide_delegate_modes():
    for mode in ("delegate", "vllm", "auto", "enforce"):
        d = decide_route(
            model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
            batch_size=1,
            greedy=True,
            fast_mode_value=mode,
        )
        assert d.route == "delegate", mode


def test_delegate_completions_parses_response():
    model = _MiniModel()
    input_ids = torch.tensor([[1, 2, 3]])

    class _Tok:
        def decode(self, ids, **kw):
            assert ids == [1, 2, 3]
            return "Hello"

        def encode(self, text, **kw):
            assert text == " world"
            return [4, 5]

    fake_body = json.dumps({"choices": [{"text": " world"}]}).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return fake_body

    decision = RouteDecision("delegate", "unsloth/Meta-Llama-3.1-8B-Instruct", "test")

    with patch("tenscuda.vllm_delegate._tokenizer", return_value=_Tok()):
        with patch("urllib.request.urlopen", return_value=_Resp()):
            out = delegate_greedy_generate(
                model,
                input_ids,
                max_new_tokens=8,
                eos_token_id=None,
                decision=decision,
            )
    assert out.tolist() == [[1, 2, 3, 4, 5]]


def test_probe_vllm_false_on_error():
    with patch("tenscuda.vllm_delegate._http_json", side_effect=VllmDelegateError("down")):
        assert probe_vllm() is False


def test_vllm_base_url_default():
    assert vllm_base_url().endswith("/v1")