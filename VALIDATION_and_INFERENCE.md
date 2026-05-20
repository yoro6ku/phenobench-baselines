# PhenoBench Validation And Inference

Use **validation** for PhenoBench `val` or `test` images when ground truth
exists and you want metrics. Use **inference** for a plain folder of new,
unlabeled images when you only want predictions and visualizations.

## Setup

```bash
cd /home/hhadhri/Bureau/code/phenobench-baselines
source /home/hhadhri/Bureau/code/phenobench/.venv/bin/activate

DATA=/home/hhadhri/Bureau/data/PhenoBench
IMAGE=06-05_00114_P0037822.png
NEW_IMAGES=/home/hhadhri/Bureau/data/Many_deseases/train/Tomato_leaf_mosaic_virus
```

Status tags:

- `tested`: ran successfully in this setup.
- `tested-1-image`: one-image smoke test passed.
- `tested-full-val`: full VAL inference and metrics were run.
- `tested-caution`: the wrapper runs, but the output has important caveats.
- `needs-test`: wrapper exists, but we have not confirmed it recently.
- `blocked`: command intentionally stops or needs external code.

If a checkpoint is missing, use the download URL in the model table. For direct
IPB-hosted weights, the wrapper can also try:

```bash
scripts/models/<wrapper>.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --download-weights
```

## How To Run

### One PhenoBench VAL Image

Use this for a quick validation smoke test and visualization:

```bash
scripts/models/<wrapper>.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1
```

Add `--open-vscode` if you want VS Code to open the saved visualization.

### Full PhenoBench VAL

Run inference on all VAL images:

```bash
scripts/models/<wrapper>.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --viz-limit 10000
```

Then run the official PhenoBench metric for that task:

```bash
phenobench-eval \
  --task <task> \
  --phenobench_dir "$DATA" \
  --prediction_dir /home/hhadhri/Bureau/code/phenobench-baselines/outputs/<model>/val/predictions \
  --split val
```

Task names are:

```text
semantics
panoptic
leaf_instances
plant_detection
leaf_detection
hierarchical
```

### New Image Folder

Use this for unlabeled images. It prepares a temporary PhenoBench-like folder,
runs the selected model, and saves predictions plus visualizations.

```bash
scripts/infer_image_folder.py "$NEW_IMAGES" \
  --model <model> \
  --limit 5
```

For all images in the folder, remove `--limit`.

Do not run `phenobench-eval` on new unlabeled images. The dummy masks created
for inference are only there to satisfy the authors' dataloaders.

Outputs are saved in both places:

```text
outputs/image_folders/<folder>_<date>/<model>/val/predictions/
outputs/image_folders/<folder>_<date>/<model>/val/visualizations/
output_<date>/<model>/<folder>/predictions/
output_<date>/<model>/<folder>/visualizations/
```

## Models From The Repo

The list is sorted from simplest output to most complete output.

### 1. Bounding Boxes

These models detect boxes only. They are useful for counting/localization, but
not for your final plant or leaf mask workflow.

| Model | Wrapper | Status | Checkpoint | Download |
| --- | --- | --- | --- | --- |
| YOLOv7 plant boxes | `infer_plant_detection_yolov7.sh` | `tested` | `plant_detection/yolov7/src/weights/yolov7_plant_detection.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/YOLOv7/yolov7_plant_detection.pt |
| YOLOv7 leaf boxes | `infer_leaf_detection_yolov7.sh` | `tested` | `leaf_detection/yolov7/src/weights/yolov7_leaf_detection.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/YOLOv7/yolov7_leaf_detection.pt |
| Faster R-CNN plant boxes | `infer_plant_detection_fasterrcnn.sh` | `needs-test` | `plant_detection/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/plant_detection/last.pt |
| Faster R-CNN leaf boxes | `infer_leaf_detection_fasterrcnn.sh` | `needs-test` | `leaf_detection/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt |
| Mask R-CNN plant boxes | `infer_plant_detection_maskrcnn.sh` | `blocked` | not separately released | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/plant_detection/last.pt |
| Mask R-CNN leaf boxes | `infer_leaf_detection_maskrcnn.sh` | `blocked` | not separately released | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt |

Author signal: these are detection baselines, not segmentation baselines.

My note: the two Mask R-CNN detection wrappers intentionally fail instead of
silently running Faster R-CNN. The released detection R-CNN code/checkpoints in
this repo are box detectors.

Example:

```bash
scripts/models/infer_leaf_detection_yolov7.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1
```

### 2. Semantic Crop/Weed Masks

These predict crop/weed/background pixels. They do not separate individual
plants or leaves.

| Model | Wrapper | Status | Checkpoint | Download |
| --- | --- | --- | --- | --- |
| ERFNet semantic segmentation | `infer_semantic_erfnet.sh` | `tested` | `semantic_segmentation/weights/semantic-seg-erfnet.ckpt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-erfnet.ckpt |
| DeepLabV3+ semantic segmentation | `infer_semantic_deeplab.sh` | `tested` | `semantic_segmentation/weights/semantic-seg-deeplab.ckpt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-deeplab.ckpt |

Author signal: these are the semantic segmentation baselines.

My note: useful if you only need crop/weed area masks. Not enough if you need
individual plant masks or leaves.

Metric:

```bash
phenobench-eval \
  --task semantics \
  --phenobench_dir "$DATA" \
  --prediction_dir /home/hhadhri/Bureau/code/phenobench-baselines/outputs/semantic_deeplab/val/predictions \
  --split val
```

### 3. Leaf Instance Masks

These separate individual leaves. This is the first useful level for your
leaf-level phytopathology workflow.

| Model | Wrapper | Status | Checkpoint | Download |
| --- | --- | --- | --- | --- |
| Mask R-CNN leaf instances | `infer_leaf_instances_maskrcnn.sh` | `tested-1-image` | `leaf_instance_segmentation/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_instance_segmentation/last.pt |
| Mask2Former leaf instances | `infer_leaf_instances_mask2former.sh` | `tested-1-image` | `outputs/docker_weights/mask2former_leaves/model_best.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/Mask2former/model.pth |

Author signal: in the paper, Mask R-CNN is slightly better for leaf instance
segmentation than Mask2Former.

My note: test both on your disease images. Mask R-CNN is the safer reported
leaf baseline, while Mask2Former is newer and may generalize differently.

Example:

```bash
scripts/models/infer_leaf_instances_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1
```

Folder inference:

```bash
scripts/infer_image_folder.py "$NEW_IMAGES" \
  --model leaf_instances_maskrcnn \
  --limit 5
```

Metric:

```bash
phenobench-eval \
  --task leaf_instances \
  --phenobench_dir "$DATA" \
  --prediction_dir /home/hhadhri/Bureau/code/phenobench-baselines/outputs/leaf_instances_maskrcnn/val/predictions \
  --split val
```

### 4. Plant Panoptic Masks

These predict semantic crop/weed classes and individual plant instances. This
is the best current level for plant-only segmentation.

| Model | Wrapper | Status | Checkpoint | Download |
| --- | --- | --- | --- | --- |
| Mask R-CNN plant panoptic | `infer_panoptic_maskrcnn.sh` | `tested-full-val` | `panoptic_segmentation/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/panoptic_segmentation/last.pt |
| Panoptic-DeepLab plant panoptic | `infer_panoptic_panopticdeeplab.sh` | `needs-test` | `outputs/docker_weights/panoptic_deeplab_plants/best.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/PanopticDeeplab/model.pth |
| Mask2Former plant panoptic | `infer_panoptic_mask2former.sh` | `tested-1-image` | `outputs/docker_weights/mask2former_plants/model_best.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/Mask2former/model.pth |

Author signal: for plant panoptic segmentation, the paper reports Mask2Former
as the strongest of these three. Reported order is Mask2Former, then Mask
R-CNN, then Panoptic-DeepLab.

My note: use `panoptic_mask2former` as the strongest plant-only baseline, and
keep `panoptic_maskrcnn` as the stable practical baseline that already ran on
full VAL.

One-image Mask2Former:

```bash
scripts/models/infer_panoptic_mask2former.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1
```

Full VAL for the stable Mask R-CNN baseline:

```bash
scripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --viz-limit 10000
```

Metric:

```bash
phenobench-eval \
  --task panoptic \
  --phenobench_dir "$DATA" \
  --prediction_dir /home/hhadhri/Bureau/code/phenobench-baselines/outputs/panoptic_maskrcnn/val/predictions \
  --split val
```

Folder inference:

```bash
scripts/infer_image_folder.py "$NEW_IMAGES" \
  --model panoptic_mask2former \
  --limit 5
```

### 5. Hierarchical Plant And Leaf Masks

These aim to predict plant instances and leaf instances together. This is the
most relevant direction for a plant pathology application, but it is also the
least simple to run.

| Model | Wrapper | Status | Checkpoint | Download |
| --- | --- | --- | --- | --- |
| Weyler hierarchical | `infer_hierarchical_weyler.sh` | `tested-caution` | `hiearchical_panoptic_segmentation/weyler/weights/weyler_checkpoint_0381.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/weyler/weyler_checkpoint_0381.pth |
| HAPT / Roggiolani hierarchical | `infer_hierarchical_hapt.sh` | `blocked` | `hiearchical_panoptic_segmentation/HAPT/weights/hapt_model.ckpt` | https://drive.google.com/drive/folders/1BctpWMAALU0l6pTvo1e6Mxs8PWplNioT?usp=sharing |

Author signal: HAPT/Roggiolani is the target hierarchical model for the
PhenoBench hierarchical task. Weyler is provided as another hierarchical
baseline, but it is not the model you identified as the final target.

My note: Weyler can look worse than `panoptic_maskrcnn` for plant masks because
the released code is crop-focused and reconstructs plant instances from leaf
parts. HAPT is the model to make work next for true plant + leaf hierarchy, but
the external `PRBonn/HAPT` code is not vendored in this repo.

Weyler one-image:

```bash
scripts/models/infer_hierarchical_weyler.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1
```

Weyler folder inference:

```bash
scripts/infer_image_folder.py "$NEW_IMAGES" \
  --model hierarchical_weyler \
  --resize 512 \
  --limit 5
```

Metric:

```bash
phenobench-eval \
  --task hierarchical \
  --phenobench_dir "$DATA" \
  --prediction_dir /home/hhadhri/Bureau/code/phenobench-baselines/outputs/hierarchical_weyler/val/predictions \
  --split val
```

## Docker Models

These wrappers require Docker with GPU access:

```text
panoptic_mask2former
leaf_instances_mask2former
panoptic_panopticdeeplab
```

Quick Docker check:

```bash
docker run --rm --gpus all mask2former_docker \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected result on this machine:

```text
True
NVIDIA RTX A2000 Laptop GPU
```

## Recommended Next Tests

1. Run `leaf_instances_maskrcnn` and `leaf_instances_mask2former` on 10 to 20
   VAL images.
2. Run `panoptic_mask2former` on more VAL images, then full VAL if it remains
   stable.
3. Run the same selected models on your disease image folders.
4. Make HAPT work when you are ready to tackle the true hierarchical model.
