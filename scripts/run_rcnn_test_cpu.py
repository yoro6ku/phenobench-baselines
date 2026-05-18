#!/usr/bin/env python3
"""Run a PhenoBench R-CNN test script with an optional CPU shim."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def patch_cuda_to_cpu() -> None:
    import torch
    import torch.nn as nn

    original_load = torch.load

    def load_on_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original_load(*args, **kwargs)

    def tensor_cuda(self, *args, **kwargs):
        return self

    def module_cuda(self, *args, **kwargs):
        return self

    torch.load = load_on_cpu
    torch.Tensor.cuda = tensor_cuda
    nn.Module.cuda = module_cuda


def patch_torchvision_detection_defaults() -> None:
    import torchvision.models.detection as detection

    def wrap_builder(builder):
        def compatible_builder(*args, **kwargs):
            if kwargs.get("pretrained") is None:
                kwargs.pop("pretrained", None)
            kwargs.setdefault("weights", None)
            kwargs.setdefault("weights_backbone", None)
            return builder(*args, **kwargs)

        return compatible_builder

    detection.fasterrcnn_resnet50_fpn = wrap_builder(detection.fasterrcnn_resnet50_fpn)
    detection.maskrcnn_resnet50_fpn = wrap_builder(detection.maskrcnn_resnet50_fpn)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_rcnn_test_cpu.py <test.py> [args...]")

    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    patch_torchvision_detection_defaults()
    if os.environ.get("PHENOBENCH_FORCE_CPU") == "1":
        patch_cuda_to_cpu()

    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
