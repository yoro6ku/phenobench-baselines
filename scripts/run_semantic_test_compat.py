#!/usr/bin/env python3
"""Run semantic test.py with small compatibility shims for old Lightning."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np


def patch_numpy_aliases() -> None:
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(np, "NINF"):
        np.NINF = -np.inf


def patch_torchmetrics_jaccard() -> None:
    import torchmetrics

    original = torchmetrics.JaccardIndex

    def compat_jaccard_index(*args, **kwargs):
        reduction = kwargs.pop("reduction", "macro")
        if reduction is None:
            kwargs.setdefault("average", "none")
        return original(*args, **kwargs)

    torchmetrics.JaccardIndex = compat_jaccard_index


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_semantic_test_compat.py <test.py> [args...]")

    patch_numpy_aliases()
    patch_torchmetrics_jaccard()
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
