"""python -m tenscuda script.py [args...]  ->  run an unmodified script with TensCUDA active.
python -m tenscuda bench [--model M] [--quick]  ->  standard benchmark (see tenscuda/bench.py)."""
import runpy
import sys

import tenscuda  # noqa: F401  (installs the patches)

if len(sys.argv) < 2:
    sys.exit("usage: python -m tenscuda script.py [args...]\n       python -m tenscuda bench [--model M] [--quick]")
if sys.argv[1] == "bench":
    from tenscuda.bench import main

    sys.exit(main(sys.argv[2:]))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
