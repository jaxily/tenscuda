"""python -m tenscuda script.py [args...]  ->  run an unmodified script with TensCUDA active.
python -m tenscuda bench [--model M] [--quick]  ->  standard benchmark (see tenscuda/bench.py).
python -m tenscuda aggregate  ->  concurrent vLLM-TT throughput (see tenscuda/aggregate.py)."""
import runpy
import sys

if len(sys.argv) < 2:
    sys.exit(
        "usage: python -m tenscuda script.py [args...]\n"
        "       python -m tenscuda bench [--model M] [--quick]\n"
        "       python -m tenscuda aggregate [--model M]"
    )
if sys.argv[1] == "bench":
    import tenscuda  # noqa: F401 — compat bench needs CUDA shim

    from tenscuda.bench import main

    sys.exit(main(sys.argv[2:]))
if sys.argv[1] == "aggregate":
    # HTTP-only client: do not open TT-XLA / CHIP_IN_USE while vLLM holds the card.
    from tenscuda.aggregate import main as aggregate_main

    sys.exit(aggregate_main(sys.argv[2:]))
import tenscuda  # noqa: F401  (installs the patches)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
