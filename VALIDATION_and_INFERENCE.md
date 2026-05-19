# PhenoBench Validation And Inference

Use **validation** when images have PhenoBench-style ground truth and you want
metrics. Use **inference** when images are unlabeled and you only want predicted
masks, boxes, and visualizations.

## Setup

```bash
cd /home/hhadhri/Bureau/code/phenobench-baselines
source /home/hhadhri/Bureau/code/phenobench/.venv/bin/activate

DATA=/home/hhadhri/Bureau/data/PhenoBench
IMAGE=06-05_00114_P0037822.png
```

## Available Models

The order goes from simpler outputs to more complete outputs.

| Output | Wrapper | Checkpoint | URL |
| --- | --- | --- | --- |
| Semantic masks | `infer_semantic_erfnet.sh` | `semantic_segmentation/weights/semantic-seg-erfnet.ckpt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-erfnet.ckpt |
| Semantic masks | `infer_semantic_deeplab.sh` | `semantic_segmentation/weights/semantic-seg-deeplab.ckpt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-deeplab.ckpt |
| Plant boxes | `infer_plant_detection_yolov7.sh` | `plant_detection/yolov7/src/weights/yolov7_plant_detection.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/YOLOv7/yolov7_plant_detection.pt |
| Leaf boxes | `infer_leaf_detection_yolov7.sh` | `leaf_detection/yolov7/src/weights/yolov7_leaf_detection.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/YOLOv7/yolov7_leaf_detection.pt |
| Plant boxes | `infer_plant_detection_fasterrcnn.sh` | `plant_detection/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/plant_detection/last.pt |
| Leaf boxes | `infer_leaf_detection_fasterrcnn.sh` | `leaf_detection/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt |
| Leaf instances | `infer_leaf_instances_maskrcnn.sh` | `leaf_instance_segmentation/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_instance_segmentation/last.pt |
| Leaf instances | `infer_leaf_instances_mask2former.sh` | `outputs/docker_weights/mask2former_leaves/model_best.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/Mask2former/model.pth |
| Plant panoptic | `infer_panoptic_maskrcnn.sh` | `panoptic_segmentation/rcnn/weights/last.pt` | https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/panoptic_segmentation/last.pt |
| Plant panoptic | `infer_panoptic_panopticdeeplab.sh` | `outputs/docker_weights/panoptic_deeplab_plants/best.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/PanopticDeeplab/model.pth |
| Plant panoptic | `infer_panoptic_mask2former.sh` | `outputs/docker_weights/mask2former_plants/model_best.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/Mask2former/model.pth |
| Hierarchical plant + leaf | `infer_hierarchical_weyler.sh` | `hiearchical_panoptic_segmentation/weyler/weights/weyler_checkpoint_0381.pth` | https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/weyler/weyler_checkpoint_0381.pth |
| Hierarchical plant + leaf | `infer_hierarchical_hapt.sh` | `hiearchical_panoptic_segmentation/HAPT/weights/hapt_model.ckpt` | https://drive.google.com/drive/folders/1BctpWMAALU0l6pTvo1e6Mxs8PWplNioT?usp=sharing |

Notes:

- Mask2Former and Panoptic-DeepLab wrappers require Docker.
- Hierarchical Weyler/HAPT are listed for completeness but are not automated
  from this repo yet.
- Separate plant/leaf detection Mask R-CNN checkpoints were not released; the
  released detection R-CNN checkpoints are Faster R-CNN checkpoints.

## One-Image Validation

Use this pattern for any working wrapper:

```bash
scripts/models/<wrapper> \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

Example for the chosen model:

```bash
scripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --image "$IMAGE" \
  --viz-limit 1 \
  --open-vscode
```

## Full VAL Validation For Panoptic Mask R-CNN

Run inference and save visualizations:

```bash
scripts/models/infer_panoptic_maskrcnn.sh \
  --phenobench-dir "$DATA" \
  --split val \
  --viz-limit 10000
```

Then compute metrics:

```bash
cd /home/hhadhri/Bureau/code/phenobench
source .venv/bin/activate

phenobench-eval \
  --task panoptic \
  --phenobench_dir /home/hhadhri/Bureau/data/PhenoBench \
  --prediction_dir /home/hhadhri/Bureau/code/phenobench-baselines/outputs/panoptic_maskrcnn/val/predictions \
  --split val
```

## Inference On A Folder Of New Images

For unlabeled images, use the folder script. It creates a temporary
PhenoBench-like input folder with dummy masks, runs the selected wrapper, and
writes predictions under `outputs/image_folders/...`. It also copies the final
results into `output_YYYY-MM-DD/<model>/<folder_name>/`.

Example:

```bash
cd /home/hhadhri/Bureau/code/phenobench-baselines
source /home/hhadhri/Bureau/code/phenobench/.venv/bin/activate

scripts/infer_image_folder.py \
  /home/hhadhri/Bureau/data/Many_deseases/train/Tomato_leaf_mosaic_virus \
  --model panoptic_maskrcnn \
  --open-vscode
```

Main outputs:

```text
outputs/image_folders/Tomato_leaf_mosaic_virus_YYYY-MM-DD/panoptic_maskrcnn/val/predictions/
outputs/image_folders/Tomato_leaf_mosaic_virus_YYYY-MM-DD/panoptic_maskrcnn/val/visualizations/
output_YYYY-MM-DD/panoptic_maskrcnn/Tomato_leaf_mosaic_virus/predictions/
output_YYYY-MM-DD/panoptic_maskrcnn/Tomato_leaf_mosaic_virus/visualizations/
```

Useful options:

```bash
# quick smoke test on the first 5 images
scripts/infer_image_folder.py /path/to/images --limit 5 --open-vscode

# choose an explicit output root
scripts/infer_image_folder.py /path/to/images \
  --output-root /home/hhadhri/Bureau/code/phenobench-baselines/outputs/my_run

# keep the generated PhenoBench-like input folder for inspection
scripts/infer_image_folder.py /path/to/images \
  --prepared-root /home/hhadhri/Bureau/data/my_images_phenobench

# skip the extra output_YYYY-MM-DD archive copy
scripts/infer_image_folder.py /path/to/images --no-daily-output
```

Do not use `phenobench-eval` on new unlabeled images. It needs real ground
truth.
