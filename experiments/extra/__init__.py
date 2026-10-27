"""Smaller analyses (paper Sections 4.2 and 4.7, Appendix B), one sub-module per analysis.

  python -m experiments.extra                 # list the analyses
  python -m experiments.extra <name> --help   # commands of one analysis

Each sub-module defines main(argv) and OUTPUTS like the main experiment modules; the outputs of all of them are
collected here (argument lists are prefixed with the sub-module name).
"""
import importlib
import pkgutil

NAMES = sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))


def _collect(attr):
    out = []
    for name in NAMES:
        mod = importlib.import_module(f"{__name__}.{name}")
        for entry in getattr(mod, attr, []):
            out.append((entry[0], [name, *entry[1]], *entry[2:]))
    return out


OUTPUTS = _collect("OUTPUTS")
