"""Greedy HF generate() via local vLLM-TT OpenAI API — skips TT-XLA decode."""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import torch

from tenscuda.fast_router import RouteDecision, vllm_env_hint

log = logging.getLogger("tenscuda.vllm_delegate")

_tokenizers: dict[str, Any] = {}


class VllmDelegateError(RuntimeError):
    """vLLM delegation failed (down, HTTP error, bad response)."""


def vllm_base_url() -> str:
    return os.environ.get("TENSCUDA_VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def vllm_served_model(model_id: str | None) -> str:
    return os.environ.get("TENSCUDA_VLLM_MODEL", model_id or "").strip()


def _timeout_sec() -> float:
    return float(os.environ.get("TENSCUDA_VLLM_TIMEOUT", "600"))


def _api_key() -> str | None:
    key = os.environ.get("TENSCUDA_VLLM_API_KEY", "").strip()
    return key or None


def _tokenizer(model_id: str):
    if model_id not in _tokenizers:
        from transformers import AutoTokenizer

        _tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _tokenizers[model_id]


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if _api_key():
        headers["Authorization"] = f"Bearer {_api_key()}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_timeout_sec()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise VllmDelegateError(f"vLLM HTTP {e.code}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise VllmDelegateError(f"vLLM unreachable at {url}: {e.reason}") from e


def probe_vllm() -> bool:
    try:
        _http_json("GET", f"{vllm_base_url()}/models")
        return True
    except VllmDelegateError:
        return False


def delegate_greedy_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id,
    decision: RouteDecision,
) -> torch.Tensor:
    """Run greedy continuation through vLLM; return HF-shaped ``input_ids`` on model device."""
    model_id = decision.model_id
    if not model_id:
        raise VllmDelegateError("delegate: missing model id on config")

    served = vllm_served_model(model_id)
    if not served:
        raise VllmDelegateError("delegate: empty TENSCUDA_VLLM_MODEL / model id")

    tok = _tokenizer(model_id)
    prompt_ids = input_ids.detach().to("cpu")[0].tolist()
    prompt = tok.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    log.info(
        "tenscuda: TENSCUDA_FAST delegate → vLLM %s (%s new tokens); skipping TT-XLA decode",
        served,
        max_new_tokens,
    )

    payload = {
        "model": served,
        "prompt": prompt,
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    data = _http_json("POST", f"{vllm_base_url()}/completions", payload)
    try:
        text = data["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise VllmDelegateError(f"unexpected vLLM response: {data!r}") from e

    new_ids = tok.encode(text, add_special_tokens=False)
    eos_set = set()
    if eos_token_id is not None:
        eos_set = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    trimmed = []
    for t in new_ids:
        if t in eos_set:
            break
        trimmed.append(t)
        if len(trimmed) >= max_new_tokens:
            break

    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = input_ids.device
    if getattr(dev, "type", None) == "meta":
        dev = input_ids.device
    prefix = input_ids.detach()
    if trimmed:
        tail = torch.tensor([trimmed], dtype=prefix.dtype, device="cpu")
        out = torch.cat([prefix.cpu(), tail], dim=1)
    else:
        out = prefix.cpu()
    return out.to(dev)


def delegate_error_hint(exc: VllmDelegateError) -> str:
    hint = vllm_env_hint()
    return (
        f"{exc}\n"
        "Start vLLM-TT on this host (metal venv), e.g.:\n"
        f"  bash scripts/run_tt_serve_fast.sh\n"
        f"  venv: {hint['venv']}\n"
        f"Set TENSCUDA_VLLM_BASE_URL if not {vllm_base_url()}."
    )