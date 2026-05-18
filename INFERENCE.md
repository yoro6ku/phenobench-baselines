# Available pretrained models 
plant_detection_yolov7       works
leaf_detection_yolov7        works
semantic_erfnet              works
semantic_deeplab             works
plant_detection_fasterrcnn   works
leaf_detection_fasterrcnn    works
panoptic_maskrcnn            works
leaf_instances_maskrcnn      works

panoptic_panopticdeeplab     needs Docker, docker not installed/on PATH
panoptic_mask2former         needs Docker, docker not installed/on PATH
leaf_instances_mask2former   needs Docker, docker not installed/on PATH
hierarchical_weyler          wrapper not automated yet, hard-code config/report paths
hierarchical_hapt            needs external PRBonn/HAPT repo


# PhenoBench Baseline Inference

This file is the single place for local one-image validation smoke tests.
The goal is to verify that each downloaded author checkpoint can load, run
inference, and produce a PhenoBench-style visualization.

## Common Setup

```bash
cd /home/hhadhri/Bureau/code/phenobench-baselines
source /home/hhadhri/Bureau/code/phenobench/.venv/bin/activate

DATA=/home/hhadhri/Bureau/data/PhenoBench
IMAGE=06-05_00114_P0037822.png
```

The test image is:

```text
/home/hhadhri/Bureau/data/PhenoBench/val/images/06-05_00114_P0037822.png
```

All wrapper outputs go under:

```text
outputs/<model_name>/val/predictions
outputs/<model_name>/val/visualizations
```

The wrapper also copies the normalized predictions and rendered visualization
into a common daily folder:

```text
output_YYYY-MM-DD/<model_name>/predictions/...
output_YYYY-MM-DD/<model_name>/visualizations/...
```

For example, a one-image YOLOv7 plant run also creates files like:

```text
output_2026-05-18/plant_detection_yolov7/visualizations/2026-05-18__plant_detection_yolov7__val__06-05_00114_P0037822__visualization.png
output_2026-05-18/plant_detection_yolov7/predictions/plant_bboxes/2026-05-18__plant_detection_yolov7__val__06-05_00114_P0037822.txt
```

Use `--no-daily-output` to disable this extra copy, or
`--daily-output-root /path/to/output_YYYY-MM-DD` to choose another common folder.

`--open-vscode` opens the rendered visualization PNG/folder in VS Code.
Remove it if you only want files written to disk.

## Downloaded Checkpoints

# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_leaf_instances_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```
```text
semantic_segmentation/weights/semantic-seg-erfnet.ckpt
semantic_segmentation/weights/semantic-seg-deeplab.ckpt

plant_detection/yolov7/src/weights/yolov7_plant_detection.pt
leaf_detection/yolov7/src/weights/yolov7_leaf_detection.pt

plant_detection/rcnn/weights/last.pt
leaf_detection/rcnn/weights/last.pt
panoptic_segmentation/rcnn/weights/last.pt
leaf_instance_segmentation/
# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_leaf_instances_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```rcnn/weights/last.pt

outputs/docker_weights/panoptic_deeplab_plants/best.pth
outputs/docker_weights/mask2former_plants/model_best.pth
outputs/docker_weights/mask2former_leaves/model_best.pth

hiearchical_panoptic_segmentation/weyler/weights/weyler_checkpoint_0381.pth
hiearchical_panoptic_segmen
# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_leaf_instances_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```tation/HAPT/weights/hapt_model.ckpt
```

## List Model Wrappers

```bash
scripts/infer_and_visualize.py --list-models
```

## Tested Status

These are the wrappers/checkpoints checked on the one-image VAL smoke test
`06-05_00114_P0037822.png`.

| Wrapper/model | Status |
| --- | --- |
# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_leaf_instances_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```
| `plant_detection_yolov7` | Works on CPU |
| `leaf_detection_yolov7` | Works on CPU |
| `semantic_erfnet` | Works on CPU |
| `semantic_deeplab` | Works on CPU |
| `plant_detection_fasterrcnn` | Works on CPU through compatibility shim |
| `leaf_detection_fasterrcnn` | Works on CPU through compatibility shim |
| `panoptic_maskrcnn` | Works on CPU through compatibility shim |
| `leaf_instances_maskrcnn` | Works on CPU through compatibility shim |
| `plant_detection_maskrcnn` | Expected error: no separate detection Mask R-CNN checkpoint/path released |
| `leaf_detection_maskrcnn` | Expected error: no separate detection Mask R-CNN checkpoint/path released |
| `panoptic_panopticdeeplab` | Not runtime-tested here: Docker is required and not on PATH |
| `panoptic_mask2former` | Not runtime-tested here: Docker is required and not on PATH |
| `leaf_instances_mask2former` | Not runtime-tested here: Docker is required and not on PATH |
| `hierarchical_weyler` | Not automated yet: authors' code hard-codes config/report paths |
| `hierarchical_hapt` | Not runnable from this repo alone: external HAPT repo required |

Checkpoint inspection note: the released plant/leaf detection R-CNN checkpoints
do not contain `network.roi_heads.mask_*` or `mask_predictor` weights. The
panoptic and leaf-instance R-CNN checkpoints do contain mask-head weights.

## 1. Plant Bounding Boxes

### YOLOv7

Uses:

```text
plant_detection/yolov7/src/weights/yolov7_plant_detection.pt
```

```bash
scripts/models/infer_plant_detection_yolov7.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### Faster R-CNN

# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_leaf_instances_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```
Uses:

```text
plant_detection/rcnn/weights/last.pt
```

```bash
# Runs through the local CPU compatibility shim by default.
scripts/models/infer_plant_detection_fasterrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### Mask R-CNN Detection

```bash
# Expected error.
# Reason: no separate plant detection Mask R-CNN inference path is released in
# this repo. The available plant detection R-CNN checkpoint has no mask-head
# weights, and plant_detection/rcnn/test.py expects FasterRCNN bbox outputs.
# This wrapper intentionally fails instead of silently running Faster R-CNN.
scripts/models/infer_plant_detection_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

## 2. Leaf Bounding Boxes

### YOLOv7

Uses:

```text
leaf_detection/yolov7/src/weights/yolov7_leaf_detection.pt
```
# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_leaf_instances_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

```bash
scripts/models/infer_leaf_detection_yolov7.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### Faster R-CNN

Uses:

```text
leaf_detection/rcnn/weights/last.pt
```

```bash
# Runs through the local CPU compatibility shim by default.
scripts/models/infer_leaf_detection_fasterrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### Mask R-CNN Detection

```bash
# Expected error.
# Reason: no separate leaf detection Mask R-CNN inference path is released in
# this repo. The available leaf detection R-CNN checkpoint has no mask-head
# weights, and leaf_detection/rcnn/test.py expects FasterRCNN bbox outputs.
# This wrapper intentionally fails instead of silently running Faster R-CNN.
scripts/models/infer_leaf_detection_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

## 3. Semantic Segmentation

### ERFNet

Uses:

```text
semantic_segmentation/weights/semantic-seg-erfnet.ckpt
```

```bash
scripts/models/infer_semantic_erfnet.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### DeepLab

Uses:

```text
semantic_segmentation/weights/semantic-seg-deeplab.ckpt
```

```bash
scripts/models/infer_semantic_deeplab.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

## 4. Leaf Instance Segmentation

### Mask R-CNN

Uses:

```text
leaf_instance_segmentation/rcnn/weights/last.pt
```

```bash
# Runs through the local CPU compatibility shim by default.
scripts/models/infer_leaf_instances_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### Mask2Former

Uses:

```text
outputs/docker_weights/mask2former_leaves/model_best.pth
```

```bash
# Warning: Docker plus NVIDIA Docker support is required.
# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_leaf_instances_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

## 5. Panoptic Segmentation

### Mask R-CNN

Uses:

```text
panoptic_segmentation/rcnn/weights/last.pt
```

```bash
# Runs through the local CPU compatibility shim by default.
scripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### Panoptic-DeepLab

Uses:

```text
outputs/docker_weights/panoptic_deeplab_plantsscripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode/best.pth
```

```bash
# Warning: Docker plus NVIDIA Docker support is required.
# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_panoptic_panopticdeeplab.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### Mask2Former

Uses:

```text
outputs/docker_weights/mask2former_plants/model_best.pth
```scripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \

```bash
# Warning: Docker plus NVIDIA Docker support is required.
# Warning: on the current machine, docker was not available when checked.
# Warning: this wrapper calls the authors' Docker/Makefile target; --image
# limits visualization selection, but may not limit Docker inference itself.
scripts/models/infer_panoptic_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

## 6. Hierarchical Segmentation
### CHOSEN MODEL V0
scripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode


### Weyler

Uses:

```text
hiearchical_panoptic_segmentation/weyler/weights/weyler_checkpoint_0381.pth
```

```bash
# Warning: this wrapper is documented but not fully automated yet.
# Reason: the authors' Weyler code hard-codes inference/report paths in
# train_config.py and report_config.py, so it still needs a config patching
# wrapper before it can run as a clean one-image smoke test.
scripts/models/infer_hierarchical_weyler.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

### HAPT

Uses:

```text
hiearchical_panoptic_segmentation/HAPT/weights/hapt_model.ckpt
```

```bash
# Warning: this will not run from this repository alone.
# Reason: the baselines repo only contains the PhenoBench dataset/config
# adapters for HAPT. The panoptic_panopticdeeplab     needs Docker, docker not installed/on PATH
# https://github.com/PRBonn/HAPT
scripts/models/infer_hierarchical_hapt.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

## Full Validation Inference

After a one-image smoke test works, remove `--image "$IMAGE"` to run the full
`val` split:

```bash
scripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --viz-limit 10000
```

## Evaluation

Use the devkit environment and point `--prediction_dir` at the wrapper output:

```bash
cd /home/hhadhri/Bureau/code/phenobench
source .venv/bin/activate

phenobench-eval \
  --task plant_detection \
  --phenobench_dir /home/hhadhri/Bureau/data/PhenoBench \
  --prediction_dir /home/hhadhri/Bureau/code/phenobench-baselines/outputs/plant_detection_yolov7/val/predictions \
  --split val
```

Change `--task` and `--prediction_dir` for each model family.
