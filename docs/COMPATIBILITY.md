# Model compatibility matrix

One P150a, single card. Greedy decoding, 32 new tokens, batch 1, bf16 weights unless noted.
"Parity" compares greedy tokens against the same prompt on the CPU. "Warm tok/s" is decode
speed after compilation; "first-run s" is the first `generate` call, including compile.
Rows are added by `scripts/matrix.py` (MATRIX-ROWS below); notes are edited by hand.

| model | params | arch | status | parity vs CPU | warm tok/s | first-run s | peak host RSS | device mem | notes |
|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-135M-Instruct | 135M | llama | works | 32/32 vs CPU fp32 | 3.6 | 50 | 1.1 GiB | n/a (1) | Host-bound; CPU runs it at 16.7 tok/s |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | llama | works | 32/32 vs CPU fp32 and bf16 | 4.2 (3.9-4.5) | 41 (36-57) | 0.68 GiB after load, 2.5 GiB end | n/a (1) | Streaming loader; CPU bf16 runs it at 5.5 tok/s |
<!-- MATRIX-ROWS -->

(1) `xm.get_memory_info` is UNIMPLEMENTED in pjrt-plugin-tt 1.5.0.dev20260831; bf16 weights alone are about 0.27 GB and 2.2 GB respectively.
