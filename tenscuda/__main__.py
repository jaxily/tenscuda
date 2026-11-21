"""python -m tenscuda script.py [args...]  ->  run an unmodified script with TensCUDA active."""
import runpy
import sys

import tenscuda  # noqa: F401  (installs the patches)

if len(sys.argv) < 2:
    sys.exit("usage: python -m tenscuda script.py [args...]")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
