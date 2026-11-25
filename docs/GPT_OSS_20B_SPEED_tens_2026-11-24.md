# gpt-oss-20b speed on `tens` — 2026-11-24

Host: **tens** · 1× Blackhole **P150** · Ryzen 7 9700X / ~32 GiB

## Stack (what actually ran)

```
tt serve jashansinghTT/gpt-oss-20b-blackhole-fast --profile p150
        │
        ▼
tt-model Docker  (image: tt-model/gpt-oss-20b-fast:…-dp)
        │
        ▼
vLLM 0.26.0.dev4 + TT platform plugin (vLLM-TT / vllm-tt-plugin)
        │
        ▼
tt-metal / ttnn on /dev/tenstorrent/0
```

| Layer | Used? | Notes |
|-------|-------|--------|
| **vLLM-TT** (`tt serve` / tt-model) | **Yes** | OpenAI API at `http://127.0.0.1:20000/v1` |
| **tt-metal** | **Yes (under vLLM)** | Device ops / warmup logs are ttnn |
| Bare tt-metal Python demo | No | Not the path under test |
| TensCUDA / TT-XLA | No | Separate compat stack (~11 tok/s class for 8B) |

Model id served: `openai/gpt-oss-20b` (weights `openai/gpt-oss-20b@6cee5e81`).

## Is this the fastest?

**Fastest known *serve bundle* for this model on P150:** yes — we used Tenstorrent’s **`blackhole-fast`** profile (`gpt-oss-20b-blackhole-fast` / `gpt-oss-20b-fast` image), not a slow/debug build.

**Peak tok/s claim:** **not proven.** Numbers below are GraphRAG `/v1/chat/completions` at **concurrency 1**, with Harmony reasoning tokens included in completion counts. We did **not** run a dedicated decode microbench, offline `vllm bench`, or a non-fast bundle A/B.

| Claim | Status |
|-------|--------|
| Vendor fast serve path | Yes (`blackhole-fast`) |
| Peak single-user decode | Unknown (not measured) |
| Multi-user aggregate | **Failed** — concurrency ≥4 crashed EngineCore |
| PR guess “~100 tok/s/user” | **Not hit** on these GraphRAG prompts |

## Measured (GraphRAG `ragmd_graphrag benchmark`)

Concurrency **1**, `max_tokens=4096`, model `openai/gpt-oss-20b` @ `:20000`.

| Role | Samples | OK | p50 tok/s | Overall tok/s | p50 latency |
|------|---------|----|-----------|---------------|-------------|
| Extract | 4 | 3 | **~37.4** | ~22.3 | ~70 s |
| Summary | 4 | 4 | **~41.5** | ~41.3 | ~32 s |

Tok/s = completion tokens / wall (includes Harmony analysis tokens). Useful JSON is a fraction of that.

## Stability

- Concurrency **1**: usable.
- Concurrency **≥4**: EngineCore internal error → container exit; required `tt serve` restart.
- Long GraphRAG extracts: timeouts / truncated JSON when `max_tokens` too low or client timeout 180 s.

## Reproduce

```bash
# on tens — model already via tt serve
curl -s http://127.0.0.1:20000/v1/models

cd ~/ragmd/scripts/graphrag-local && source .venv/bin/activate
export EXTRACT_MODEL=openai/gpt-oss-20b EXTRACT_BASE_URL=http://127.0.0.1:20000/v1
export SUMMARY_MODEL=openai/gpt-oss-20b SUMMARY_BASE_URL=http://127.0.0.1:20000/v1
export GRAPHRAG_MAX_TOKENS=4096 GRAPHRAG_LLM_TIMEOUT_S=900
python -m ragmd_graphrag benchmark --samples 4 --concurrency 1
```

Stop: `tt model stop jashansinghTT/gpt-oss-20b-blackhole-fast`
