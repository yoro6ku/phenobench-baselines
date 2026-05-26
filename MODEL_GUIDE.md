# PhenoBench Model Guide

Short, practical explanation of the model families we are validating.

## Timing

`scripts/run_validation_suite.sh` reports:

- `inference_seconds`, `inference_seconds_per_image`, `fps`: model execution only, before our visualization and PhenoBench evaluation.
- `pipeline_seconds`: wrapper time, including normalization and visualization.

For Docker, Weyler, and HAPT, inference time is the author command needed to emit usable predictions, so their own internal postprocessing may be included.

## Models

### YOLOv7

Fast detector. It predicts rectangles around plants or leaves.

Output: boxes only. Useful for quick localization, not for disease mask work.

### ERFNet / DeepLab

Semantic segmentation. They paint every pixel as soil, crop, weed, etc.

Output: class mask only. Good for crop/weed/background, but no individual plant or leaf IDs.

### Faster R-CNN

Two-stage detector. It proposes object regions, then refines boxes.

Output: boxes only. Slower and more structured than YOLO, but still no masks.

### Mask R-CNN

Faster R-CNN plus a mask head.

Output: instance masks. In this repo we use it for plant panoptic segmentation and leaf instance segmentation. It is the reliable baseline we reproduced well.

### Panoptic-DeepLab

Dense panoptic segmentation. It predicts semantic pixels plus instance grouping cues.

Output: plant panoptic masks. Strong plant-level model, but not leaf hierarchy.

### Mask2Former

Transformer-style segmentation. It predicts a set of masks using attention.

Output: plant panoptic masks or leaf instance masks, depending on checkpoint. Modern and strong, but Docker/GPU-based here.

### Weyler

Hierarchical-style crop plant and leaf instance model.

Output: plant instances and leaf instances. Useful to test, but it is not the main Roggiolani HAPT method.

### HAPT

Hierarchical model from Roggiolani et al.

Output: semantic classes -> plant instances -> leaf instances. It does not simply run the baseline models one after another; the hierarchy is part of the model design.

Best conceptual fit for plant pathology, but the public code/checkpoint path is more fragile and can be memory-heavy on a 4 GB GPU.
