"""Greedy HF generate() via local vLLM-TT OpenAI API — skips TT-XLA decode."""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import torch

from tenscuda.fast_router import RouteDecision, default_aggregate_concurrency, vllm_env_hint

log = logging.getLogger("tenscuda.vllm_delegate")

_tokenizers: dict[str, Any] = {}


class VllmDelegateError(RuntimeError):
    """vLLM delegation failed (down, HTTP error, bad response)."""


@dataclass(frozen=True)
class AggregateResult:
    """Wall-clock aggregate throughput from concurrent vLLM completions."""

    model_id: str
    served_model: str
    num_prompts: int
    concurrency: int
    max_new_tokens: int
    wall_seconds: float
    total_new_tokens: int
    errors: tuple[str, ...]

    @property
    def aggregate_tok_s(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.total_new_tokens / self.wall_seconds


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


def _completion_new_token_count(data: dict) -> int:
    usage = data.get("usage") or {}
    if isinstance(usage, dict):
        for key in ("completion_tokens", "output_tokens"):
            if key in usage:
                return int(usage[key])
    try:
        text = data["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return 0
    return max(0, len(text.split()))


def _one_completion(
    served: str,
    prompt: str,
    max_new_tokens: int,
) -> tuple[int, str | None]:
    payload = {
        "model": served,
        "prompt": prompt,
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    try:
        data = _http_json("POST", f"{vllm_base_url()}/completions", payload)
        return _completion_new_token_count(data), None
    except VllmDelegateError as e:
        return 0, str(e)


def delegate_aggregate(
    *,
    model_id: str,
    prompts: list[str] | None = None,
    num_prompts: int | None = None,
    max_new_tokens: int = 128,
    concurrency: int | None = None,
    default_prompt: str = "Explain in two sentences why the sky is blue.",
) -> AggregateResult:
    """Saturate vLLM-TT with concurrent ``/v1/completions`` (target ~900+ tok/s aggregate)."""
    if not model_id:
        raise VllmDelegateError("aggregate: missing model_id")
    served = vllm_served_model(model_id)
    if not served:
        raise VllmDelegateError("aggregate: empty TENSCUDA_VLLM_MODEL / model id")

    conc = concurrency if concurrency is not None else default_aggregate_concurrency()
    conc = max(1, int(conc))
    if prompts is None:
        n = num_prompts if num_prompts is not None else conc
        n = max(1, int(n))
        prompts = [f"{default_prompt} (stream {i})" for i in range(n)]
    if not prompts:
        raise VllmDelegateError("aggregate: no prompts")

    log.info(
        "tenscuda: aggregate vLLM %s — %d prompts, concurrency=%d, max_new_tokens=%d",
        served,
        len(prompts),
        conc,
        max_new_tokens,
    )

    total_new = 0
    errors: list[str] = []
    t0 = time.perf_counter()
    workers = min(conc, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_one_completion, served, prompt, max_new_tokens) for prompt in prompts
        ]
        for fut in as_completed(futures):
            n_tok, err = fut.result()
            total_new += n_tok
            if err:
                errors.append(err)
    wall = time.perf_counter() - t0

    if errors and total_new == 0:
        raise VllmDelegateError(
            f"aggregate: all {len(errors)} requests failed; first: {errors[0]}"
        )

    return AggregateResult(
        model_id=model_id,
        served_model=served,
        num_prompts=len(prompts),
        concurrency=conc,
        max_new_tokens=max_new_tokens,
        wall_seconds=wall,
        total_new_tokens=total_new,
        errors=tuple(errors),
    )


def _greedy_tail_ids(tok, text: str, eos_token_id, max_new_tokens: int) -> list[int]:
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
    return trimmed


def _completion_to_ids(tok, data: dict, eos_token_id, max_new_tokens: int) -> list[int]:
    try:
        text = data["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise VllmDelegateError(f"unexpected vLLM response: {data!r}") from e
    return _greedy_tail_ids(tok, text, eos_token_id, max_new_tokens)


def delegate_batch_greedy_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id,
    decision: RouteDecision,
) -> torch.Tensor:
    """Greedy decode for batch>1 via concurrent vLLM completions (one HTTP call per row)."""
    model_id = decision.model_id
    if not model_id:
        raise VllmDelegateError("delegate batch: missing model id on config")
    served = vllm_served_model(model_id)
    if not served:
        raise VllmDelegateError("delegate batch: empty TENSCUDA_VLLM_MODEL / model id")

    tok = _tokenizer(model_id)
    rows = input_ids.detach().to("cpu")
    batch = int(rows.shape[0])
    log.info(
        "tenscuda: TENSCUDA_FAST delegate batch=%d → vLLM %s; skipping TT-XLA",
        batch,
        served,
    )

    def _row_request(row_idx: int) -> list[int]:
        prompt_ids = rows[row_idx].tolist()
        prompt = tok.decode(
            prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
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
        return _completion_to_ids(tok, data, eos_token_id, max_new_tokens)

    tails: list[list[int]] = [[] for _ in range(batch)]
    workers = min(batch, default_aggregate_concurrency())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_row_request, i): i for i in range(batch)}
        for fut in as_completed(futs):
            idx = futs[fut]
            tails[idx] = fut.result()

    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = input_ids.device
    if getattr(dev, "type", None) == "meta":
        dev = input_ids.device

    out_rows = []
    for i in range(batch):
        prefix = rows[i : i + 1]
        if tails[i]:
            tail = torch.tensor([tails[i]], dtype=prefix.dtype, device="cpu")
            out_rows.append(torch.cat([prefix, tail], dim=1))
        else:
            out_rows.append(prefix)
    max_len = max(r.shape[1] for r in out_rows)
    padded = []
    for r in out_rows:
        if r.shape[1] < max_len:
            pad = torch.zeros(1, max_len - r.shape[1], dtype=r.dtype, device="cpu")
            r = torch.cat([r, pad], dim=1)
        padded.append(r)
    return torch.cat(padded, dim=0).to(dev)


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
    trimmed = _completion_to_ids(tok, data, eos_token_id, max_new_tokens)

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