import importlib
import sys

from experiments.extra import NAMES

if len(sys.argv) < 2 or sys.argv[1] not in NAMES:
    print("usage: python -m experiments.extra <name> ...\n\nanalyses:")
    for name in NAMES:
        doc = (importlib.import_module(f"experiments.extra.{name}").__doc__ or "").strip().split("\n")[0]
        print(f"  {name:22s} {doc}")
    sys.exit(0 if len(sys.argv) < 2 else 2)
importlib.import_module(f"experiments.extra.{sys.argv[1]}").main(sys.argv[2:])
