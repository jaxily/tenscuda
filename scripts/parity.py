"""Greedy token parity: CPU (HF generate) vs TT (tenscuda), same prompt.

    python scripts/parity.py MODEL N   (needs CPU ref json from examples/tinyllama_generate.py --out)
"""
import json, sys
ref = json.load(open(sys.argv[1]))["runs"][-1]["tokens"]
got = json.load(open(sys.argv[2]))["runs"][-1]["tokens"]
n = 0
for a, b in zip(ref, got):
    if a != b:
        break
    n += 1
print(f"matching prefix: {n}/{len(ref)} tokens; identical={ref == got}")
