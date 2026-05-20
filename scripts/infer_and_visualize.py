#!/usr/bin/env python3
"""Run a PhenoBench baseline model and render PhenoBench-style visualizations."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVKIT_ROOT = REPO_ROOT.parent / "phenobench"
DEFAULT_PYTHON = DEFAULT_DEVKIT_ROOT / ".venv" / "bin" / "python"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


MODELS: Dict[str, Dict] = {
    "semantic_erfnet": {
        "task": "semantics",
        "kind": "semantic_lightning",
        "workdir": "semantic_segmentation",
        "script": "test.py",
        "config": "config/config_erfnet.yaml",
        "weights": "semantic_segmentation/weights/semantic-seg-erfnet.ckpt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-erfnet.ckpt",
        "deps": "semantic_segmentation/setup/requirements.txt",
    },
    "semantic_deeplab": {
        "task": "semantics",
        "kind": "semantic_lightning",
        "workdir": "semantic_segmentation",
        "script": "test.py",
        "config": "config/config_deeplab.yaml",
        "weights": "semantic_segmentation/weights/semantic-seg-deeplab.ckpt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-deeplab.ckpt",
        "deps": "semantic_segmentation/setup/requirements.txt",
    },
    "plant_detection_yolov7": {
        "task": "plant_detection",
        "kind": "yolov7",
        "workdir": "plant_detection/yolov7/src",
        "weights": "plant_detection/yolov7/src/weights/yolov7_plant_detection.pt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/YOLOv7/yolov7_plant_detection.pt",
        "prediction_subdir": "plant_bboxes",
        "label_offset": 0,
        "deps": "plant_detection/yolov7/src/yolov7/requirements.txt",
    },
    "leaf_detection_yolov7": {
        "task": "leaf_detection",
        "kind": "yolov7",
        "workdir": "leaf_detection/yolov7/src",
        "weights": "leaf_detection/yolov7/src/weights/yolov7_leaf_detection.pt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/YOLOv7/yolov7_leaf_detection.pt",
        "prediction_subdir": "leaf_bboxes",
        "label_offset": 1,
        "deps": "leaf_detection/yolov7/src/yolov7/requirements.txt",
    },
    "plant_detection_fasterrcnn": {
        "task": "plant_detection",
        "kind": "rcnn",
        "workdir": "plant_detection/rcnn",
        "script": "test.py",
        "config": "configs/fasterrcnn_plants.yaml",
        "weights": "plant_detection/rcnn/weights/last.pt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/plant_detection/last.pt",
        "deps": "plant_detection/rcnn/environment.yml",
        "model_id": "fasterrcnn",
    },
    "plant_detection_maskrcnn": {
        "task": "plant_detection",
        "kind": "unsupported",
        "workdir": "plant_detection/rcnn",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/plant_detection/last.pt",
        "deps": "plant_detection/rcnn/environment.yml",
        "note": (
            "No separate plant detection Mask R-CNN inference path is released in this repo. "
            "The available plant detection R-CNN checkpoint has no mask-head weights and the "
            "plant detection test.py expects FasterRCNN bbox outputs."
        ),
    },
    "leaf_detection_fasterrcnn": {
        "task": "leaf_detection",
        "kind": "rcnn",
        "workdir": "leaf_detection/rcnn",
        "script": "test.py",
        "config": "configs/fasterrcnn_leaves.yaml",
        "weights": "leaf_detection/rcnn/weights/last.pt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt",
        "deps": "leaf_detection/rcnn/environment.yml",
        "model_id": "fasterrcnn_leaves",
    },
    "leaf_detection_maskrcnn": {
        "task": "leaf_detection",
        "kind": "unsupported",
        "workdir": "leaf_detection/rcnn",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt",
        "deps": "leaf_detection/rcnn/environment.yml",
        "note": (
            "No separate leaf detection Mask R-CNN inference path is released in this repo. "
            "The available leaf detection R-CNN checkpoint has no mask-head weights and the "
            "leaf detection test.py expects FasterRCNN bbox outputs."
        ),
    },
    "panoptic_maskrcnn": {
        "task": "panoptic",
        "kind": "rcnn",
        "workdir": "panoptic_segmentation/rcnn",
        "script": "test.py",
        "config": "configs/maskrcnn_plants.yaml",
        "weights": "panoptic_segmentation/rcnn/weights/last.pt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/panoptic_segmentation/last.pt",
        "deps": "panoptic_segmentation/rcnn/environment.yml",
        "model_id": "maskrcnn",
    },
    "leaf_instances_maskrcnn": {
        "task": "leaf_instances",
        "kind": "rcnn",
        "workdir": "leaf_instance_segmentation/rcnn",
        "script": "test.py",
        "config": "configs/maskrcnn_leaves.yaml",
        "weights": "leaf_instance_segmentation/rcnn/weights/last.pt",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_instance_segmentation/last.pt",
        "deps": "leaf_instance_segmentation/rcnn/environment.yml",
        "model_id": "maskrcnn_leaves",
    },
    "panoptic_panopticdeeplab": {
        "task": "panoptic",
        "kind": "docker_make",
        "workdir": "panoptic_segmentation/panopticdeeplab",
        "target": {"val": "predict_val_plants", "test": "predict_test_plants"},
        "weights": "outputs/docker_weights/panoptic_deeplab_plants/best.pth",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/PanopticDeeplab/model.pth",
        "docker_weight_dest": "panoptic_deeplab_plants/best.pth",
        "deps": "panoptic_segmentation/panopticdeeplab/README.md",
    },
    "panoptic_mask2former": {
        "task": "panoptic",
        "kind": "docker_make",
        "workdir": "leaf_instance_segmentation/mask2former",
        "target": {"val": "predict_val_plants", "test": "predict_test_plants"},
        "weights": "outputs/docker_weights/mask2former_plants/model_best.pth",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/Mask2former/model.pth",
        "docker_weight_dest": "mask2former_plants/model_best.pth",
        "deps": "leaf_instance_segmentation/mask2former/README.md",
    },
    "leaf_instances_mask2former": {
        "task": "leaf_instances",
        "kind": "docker_make",
        "workdir": "leaf_instance_segmentation/mask2former",
        "target": {"val": "predict_val_leaves", "test": "predict_test_leaves"},
        "weights": "outputs/docker_weights/mask2former_leaves/model_best.pth",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/Mask2former/model.pth",
        "docker_weight_dest": "mask2former_leaves/model_best.pth",
        "deps": "leaf_instance_segmentation/mask2former/README.md",
    },
    "hierarchical_weyler": {
        "task": "hierarchical",
        "kind": "weyler",
        "workdir": "hiearchical_panoptic_segmentation/weyler",
        "weights": "hiearchical_panoptic_segmentation/weyler/weights/weyler_checkpoint_0381.pth",
        "url": "https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/weyler/weyler_checkpoint_0381.pth",
        "deps": "hiearchical_panoptic_segmentation/weyler/requirements.txt",
        "note": (
            "Weyler predicts crop plant and leaf instances. Semantics are derived "
            "from predicted plant instances as crop-vs-background for normalized output."
        ),
    },
    "hierarchical_hapt": {
        "task": "hierarchical",
        "kind": "external_hapt",
        "workdir": "hiearchical_panoptic_segmentation/HAPT",
        "url": "https://drive.google.com/drive/folders/1BctpWMAALU0l6pTvo1e6Mxs8PWplNioT?usp=sharing",
        "deps": "hiearchical_panoptic_segmentation/HAPT/README.md",
        "note": "The HAPT code is not vendored here; this folder only contains the PhenoBench dataset/config adapters.",
    },
}


def rel(path: str) -> Path:
    return REPO_ROOT / path


def display_path(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def run(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None, dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def log_step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def ensure_yaml() -> None:
    if yaml is None:
        raise SystemExit("PyYAML is required for this wrapper. Install it in the Python environment running this script.")


def ensure_images_exist(phenobench_dir: Path, split: str, image_names: List[str]) -> None:
    missing = [name for name in image_names if not (phenobench_dir / split / "images" / Path(name).name).exists()]
    if missing:
        raise SystemExit(f"Missing image(s) in {phenobench_dir / split / 'images'}: {', '.join(missing)}")


def image_names_for_split(phenobench_dir: Path, split: str, requested: List[str]) -> List[str]:
    if requested:
        names = [Path(name).name for name in requested]
    else:
        names = [path.name for path in sorted((phenobench_dir / split / "images").glob("*.png"))]
    ensure_images_exist(phenobench_dir, split, names)
    return names


def symlink_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(source)
    except OSError:
        shutil.copy2(source, dest)


def populate_split_view(source_split: Path, dest_split: Path, image_names: List[str], copy_files: bool = False) -> None:
    for source_field in sorted(path for path in source_split.iterdir() if path.is_dir()):
        dest_field = dest_split / source_field.name
        dest_field.mkdir(parents=True, exist_ok=True)
        for image_name in image_names:
            source = source_field / image_name
            if source.exists():
                dest = dest_field / image_name
                if copy_files:
                    shutil.copy2(source, dest)
                else:
                    symlink_or_copy(source, dest)


@contextmanager
def semantic_dataset_view(phenobench_dir: Path, split: str, image_names: List[str]):
    """Expose the requested split as test/, because the semantic baseline always tests test/."""
    with tempfile.TemporaryDirectory(prefix="phenobench_semantic_") as temp_dir:
        root = Path(temp_dir)
        for dataset_split in ["train", "val", "test"]:
            (root / dataset_split / "images").mkdir(parents=True, exist_ok=True)
        populate_split_view(phenobench_dir / split, root / "test", image_names)
        yield root


@contextmanager
def rcnn_split_view(phenobench_dir: Path, split: str, image_names: List[str], copy_files: bool = False):
    with tempfile.TemporaryDirectory(prefix="phenobench_rcnn_") as temp_dir:
        split_dir = Path(temp_dir) / split
        populate_split_view(phenobench_dir / split, split_dir, image_names, copy_files=copy_files)
        yield Path(temp_dir)


def ensure_weight(spec: Dict, weights: Path, download: bool, dry_run: bool) -> Path:
    weights.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists():
        return weights
    if not download:
        raise SystemExit(
            f"Missing weights: {weights}\n"
            f"Download URL: {spec.get('url', 'not available')}\n"
            "Pass --download-weights or provide --weights /path/to/checkpoint."
        )
    url = spec.get("url")
    if not url or "drive.google.com" in url:
        raise SystemExit(f"Cannot auto-download this checkpoint. Download it manually from: {url}")
    if dry_run:
        print(f"Would download {url}\n  -> {weights}")
        return weights
    print(f"Downloading {url}\n  -> {weights}")
    urllib.request.urlretrieve(url, weights)
    return weights


def temp_config_for_semantic(config_path: Path, phenobench_dir: Path) -> Path:
    ensure_yaml()
    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    cfg["data"]["name"] = "PDC"
    cfg["data"]["path_to_dataset"] = str(phenobench_dir)
    cfg["data"]["num_workers"] = 0
    cfg["test"]["n_gpus"] = 0
    cfg["test"]["batch_size"] = 1
    temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, temp)
    temp.close()
    return Path(temp.name)


def temp_config_for_rcnn(config_path: Path, phenobench_dir: Path, split: str, output_dir: Path, model_id: str) -> Path:
    ensure_yaml()
    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    cfg["experiment"]["id"] = model_id
    cfg["checkpoint"] = str(output_dir / "checkpoints")
    cfg["tensorboard"] = str(output_dir / "tensorboard")
    cfg["data"]["train"] = str(phenobench_dir / "train")
    cfg["data"]["val"] = str(phenobench_dir / split)
    cfg["train"]["workers"] = 0
    cfg["train"]["batch_size"] = 1
    cfg["train"]["n_gpus"] = 0
    temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, temp)
    temp.close()
    return Path(temp.name)


def selected_images(phenobench_dir: Path, split: str, requested: List[str]) -> List[Path]:
    image_dir = phenobench_dir / split / "images"
    names = image_names_for_split(phenobench_dir, split, requested)
    return [image_dir / name for name in names]


def common_image_size(image_paths: List[Path]) -> tuple[int, int]:
    sizes = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            sizes.append(image.size)
    unique_sizes = sorted(set(sizes))
    if len(unique_sizes) != 1:
        details = ", ".join(f"{width}x{height}" for width, height in unique_sizes[:5])
        raise SystemExit(
            "Weyler hierarchical inference expects all selected images to have the same size. "
            f"Found: {details}"
        )
    return unique_sizes[0]


def ensure_weyler_size(width: int, height: int) -> None:
    if width % 8 == 0 and height % 8 == 0:
        return
    raise SystemExit(
        "Weyler hierarchical inference needs image dimensions divisible by 8. "
        f"Found {width}x{height}. For plain image folders, use "
        "`scripts/infer_image_folder.py /path/to/images --model hierarchical_weyler --resize 512`."
    )


def normalize_yolo_labels(raw_labels: Path, out_dir: Path, image_paths: List[Path], label_offset: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        source = raw_labels / f"{image_path.stem}.txt"
        dest = out_dir / f"{image_path.stem}.txt"
        if not source.exists() or source.stat().st_size == 0:
            dest.write_text("")
            continue
        rows = []
        with source.open() as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                parts[0] = str(int(float(parts[0])) + label_offset)
                rows.append(" ".join(parts))
        dest.write_text(("\n".join(rows) + "\n") if rows else "")


def normalize_known_outputs(raw_dir: Path, predictions_dir: Path) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "semantics": [
            "semantics",
            "predictions/semantics",
            "mask2former_plants/*_predictions/semantics",
            "mask2former_leaves/*_predictions/semantics",
            "panoptic_deeplab_plants/*_predictions/semantics",
            "postprocess/arg_max_class",
            "arg_max_class",
            "lightning_logs/version_*/postprocess/arg_max_class",
        ],
        "plant_instances": [
            "plant_instances",
            "predictions/plant_instances",
            "mask2former_plants/*_predictions/instances",
            "panoptic_deeplab_plants/*_predictions/plant_instances",
            "instances",
        ],
        "leaf_instances": [
            "leaf_instances",
            "predictions/leaf_instances",
            "mask2former_leaves/*_predictions/instances",
            "instances",
        ],
        "plant_bboxes": ["plant_bboxes", "predictions/plant_bboxes", "labels"],
        "leaf_bboxes": ["leaf_bboxes", "predictions/leaf_bboxes", "labels"],
    }
    for name, subdirs in candidates.items():
        dest = predictions_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        copied = False
        for subdir in subdirs:
            sources = sorted(raw_dir.glob(subdir)) if "*" in subdir else [raw_dir / subdir]
            for source in reversed(sources):
                if not source.exists() or not source.is_dir():
                    continue
                shutil.copytree(source, dest)
                copied = True
                count = sum(1 for path in dest.rglob("*") if path.is_file())
                print(f"Normalized {name}: {count} file(s) -> {dest}", flush=True)
                break
            if copied:
                break


def run_yolov7(spec: Dict, args: argparse.Namespace, output_dir: Path, weights: Path) -> Path:
    workdir = rel(spec["workdir"])
    raw_name = "raw"
    images = selected_images(args.phenobench_dir, args.split, args.image)
    if args.image and len(images) != 1:
        raise SystemExit("YOLOv7 wrapper currently supports either all images or one --image for smoke tests.")
    source = images[0] if args.image else args.phenobench_dir / args.split / "images"
    log_step(f"YOLOv7 inference on {len(images)} image(s)")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    cmd = [
        str(args.python),
        "yolov7/detect.py",
        "--weights",
        display_path(weights, workdir),
        "--source",
        str(source),
        "--img-size",
        str(args.img_size),
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.iou_thres),
        "--device",
        args.device,
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--project",
        str(output_dir),
        "--name",
        raw_name,
        "--exist-ok",
        "--no-trace",
    ]
    run(cmd, cwd=workdir, env=env, dry_run=args.dry_run)
    predictions_dir = output_dir / "predictions"
    if not args.dry_run:
        log_step("Normalizing YOLO label predictions")
        normalize_yolo_labels(
            output_dir / raw_name / "labels",
            predictions_dir / spec["prediction_subdir"],
            images,
            int(spec.get("label_offset", 0)),
        )
    return predictions_dir


def run_semantic(spec: Dict, args: argparse.Namespace, output_dir: Path, weights: Path) -> Path:
    workdir = rel(spec["workdir"])
    raw_dir = output_dir / "raw"
    image_names = image_names_for_split(args.phenobench_dir, args.split, args.image)
    log_step(f"Semantic inference on {len(image_names)} image(s)")
    with semantic_dataset_view(args.phenobench_dir, args.split, image_names) as dataset_root:
        config = temp_config_for_semantic(workdir / spec["config"], dataset_root)
        runner = REPO_ROOT / "scripts" / "run_semantic_test_compat.py"
        cmd = [
            str(args.python),
            str(runner),
            spec["script"],
            "--config",
            str(config),
            "--ckpt_path",
            str(weights),
            "--export_dir",
            str(raw_dir),
        ]
        run(cmd, cwd=workdir, dry_run=args.dry_run)
    predictions_dir = output_dir / "predictions"
    if not args.dry_run:
        log_step("Normalizing semantic predictions")
        normalize_known_outputs(raw_dir, predictions_dir)
    return predictions_dir


def run_rcnn(spec: Dict, args: argparse.Namespace, output_dir: Path, weights: Path) -> Path:
    workdir = rel(spec["workdir"])
    raw_dir = output_dir / "raw"
    image_names = image_names_for_split(args.phenobench_dir, args.split, args.image)
    log_step(f"R-CNN inference on {len(image_names)} image(s)")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if args.device == "cpu":
        env["PHENOBENCH_FORCE_CPU"] = "1"
    with rcnn_split_view(args.phenobench_dir, args.split, image_names) as dataset_root:
        config = temp_config_for_rcnn(
            workdir / spec["config"],
            dataset_root,
            args.split,
            output_dir,
            spec["model_id"],
        )
        runner = REPO_ROOT / "scripts" / "run_rcnn_test_cpu.py"
        cmd = [str(args.python), str(runner), spec["script"], "-c", str(config), "-w", str(weights), "-o", str(raw_dir)]
        run(cmd, cwd=workdir, env=env, dry_run=args.dry_run)
    predictions_dir = output_dir / "predictions"
    if not args.dry_run:
        log_step("Normalizing R-CNN predictions")
        normalize_known_outputs(raw_dir, predictions_dir)
    return predictions_dir


def normalize_weyler_outputs(raw_dir: Path, predictions_dir: Path, split: str) -> None:
    report_instances = raw_dir / "reports" / split / "-001" / "instances"
    plant_source = report_instances / "objects"
    leaf_source = report_instances / "parts"
    if not plant_source.exists() or not leaf_source.exists():
        raise SystemExit(
            "Weyler report output was not found. Expected:\n"
            f"  {plant_source}\n"
            f"  {leaf_source}"
        )

    for name, source in {"plant_instances": plant_source, "leaf_instances": leaf_source}.items():
        dest = predictions_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        count = sum(1 for path in dest.glob("*.png"))
        print(f"Normalized {name}: {count} file(s) -> {dest}", flush=True)

    semantics_dir = predictions_dir / "semantics"
    if semantics_dir.exists():
        shutil.rmtree(semantics_dir)
    semantics_dir.mkdir(parents=True, exist_ok=True)
    for plant_path in sorted((predictions_dir / "plant_instances").glob("*.png")):
        plant_instances = np.array(Image.open(plant_path))
        semantics = (plant_instances > 0).astype(np.uint8)
        Image.fromarray(semantics).save(semantics_dir / plant_path.name)
    print(f"Normalized semantics: {len(list(semantics_dir.glob('*.png')))} file(s) -> {semantics_dir}", flush=True)


def run_weyler(spec: Dict, args: argparse.Namespace, output_dir: Path, weights: Path) -> Path:
    workdir = rel(spec["workdir"])
    raw_dir = output_dir / "raw"
    image_names = image_names_for_split(args.phenobench_dir, args.split, args.image)
    image_paths = [args.phenobench_dir / args.split / "images" / name for name in image_names]
    width, height = common_image_size(image_paths)
    ensure_weyler_size(width, height)
    log_step(f"Weyler hierarchical inference on {len(image_names)} image(s)")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["DATASET_DIR"] = ""
    env["WEYLER_SPLIT"] = args.split
    env["WEYLER_LOG_DIR"] = str(raw_dir)
    env["WEYLER_SAVE_DIR"] = str(output_dir / "checkpoints")
    env["WEYLER_RESUME_PATH"] = str(weights)
    env["WEYLER_ONLY_EVAL"] = "1"
    env["WEYLER_CUDA"] = "0" if args.device == "cpu" else "1"
    env["WEYLER_WORKERS"] = "0"
    env["WEYLER_BATCH_SIZE"] = "1"
    env["WEYLER_WIDTH"] = str(width)
    env["WEYLER_HEIGHT"] = str(height)

    with rcnn_split_view(args.phenobench_dir, args.split, image_names) as dataset_root:
        env["DATASET_DIR"] = str(dataset_root)
        run([str(args.python), "src/train.py"], cwd=workdir, env=env, dry_run=args.dry_run)
        run([str(args.python), "src/report.py"], cwd=workdir, env=env, dry_run=args.dry_run)

    predictions_dir = output_dir / "predictions"
    if not args.dry_run:
        log_step("Normalizing Weyler predictions")
        normalize_weyler_outputs(raw_dir, predictions_dir, args.split)
    return predictions_dir


def prepare_docker_weights(spec: Dict, output_dir: Path, weights: Path) -> None:
    dest = output_dir / spec["docker_weight_dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not dest.is_symlink():
        return
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(weights, dest)
    except OSError:
        shutil.copy2(weights, dest)


def run_docker_make(spec: Dict, args: argparse.Namespace, output_dir: Path, weights: Path) -> Path:
    if shutil.which("docker") is None and not args.dry_run:
        raise SystemExit(
            "Docker is required for this baseline, but `docker` is not on PATH.\n"
            f"See {rel(spec['deps'])}."
        )
    target = spec["target"].get(args.split)
    if target is None:
        raise SystemExit(f"{args.model} has no Makefile target for split={args.split}.")
    if not args.dry_run:
        prepare_docker_weights(spec, output_dir, weights)
    image_names = image_names_for_split(args.phenobench_dir, args.split, args.image)
    log_step(f"Docker/Make inference for {args.model} on {len(image_names)} image(s)")
    if args.image:
        with rcnn_split_view(args.phenobench_dir, args.split, image_names, copy_files=True) as dataset_root:
            cmd = ["make", f"data_dir={dataset_root}", f"log_dir={output_dir}", target]
            run(cmd, cwd=rel(spec["workdir"]), dry_run=args.dry_run)
    else:
        cmd = ["make", f"data_dir={args.phenobench_dir}", f"log_dir={output_dir}", target]
        run(cmd, cwd=rel(spec["workdir"]), dry_run=args.dry_run)
    predictions_dir = output_dir / "predictions"
    if not args.dry_run:
        log_step("Normalizing Docker predictions")
        normalize_known_outputs(output_dir, predictions_dir)
    return predictions_dir


def unsupported(spec: Dict, args: argparse.Namespace) -> Path:
    if spec.get("kind") == "unsupported":
        message = [
            f"{args.model} is intentionally unsupported by this wrapper.",
            spec.get("note", ""),
            "This command exits here instead of silently running a different model.",
        ]
    else:
        message = [
            f"{args.model} is not fully automatable from this repository alone.",
            spec.get("note", ""),
        ]
    message.append(f"Read: {rel(spec['deps'])}")
    if spec.get("url"):
        message.append(f"Weights/code URL: {spec['url']}")
    raise SystemExit("\n".join(line for line in message if line))


def visualize(spec: Dict, args: argparse.Namespace, prediction_dir: Path, output_dir: Path) -> Optional[Path]:
    if args.no_visualize:
        return None
    visualization_dir = output_dir / "visualizations"
    log_step(f"Rendering visualizations to {visualization_dir}")
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "visualize_predictions.py"),
        "--task",
        spec["task"],
        "--phenobench-dir",
        str(args.phenobench_dir),
        "--prediction-dir",
        str(prediction_dir),
        "--output-dir",
        str(visualization_dir),
        "--split",
        args.split,
        "--limit",
        str(args.viz_limit),
        "--devkit-root",
        str(args.devkit_root),
    ]
    for image in args.image:
        cmd.extend(["--image", image])
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    run(cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)
    return visualization_dir


def prefixed_result_name(run_date: str, model: str, split: str, image_stem: str, suffix: str, label: Optional[str] = None) -> str:
    parts = [run_date, model, split, image_stem]
    if label:
        parts.append(label)
    return "__".join(parts) + suffix


def archive_image_names(args: argparse.Namespace, visualization_dir: Optional[Path]) -> List[str]:
    if args.image:
        return [Path(name).name for name in args.image]
    if visualization_dir is not None and visualization_dir.exists():
        return [path.name for path in sorted(visualization_dir.glob("*.png"))]
    return []


def archive_results(
    args: argparse.Namespace,
    prediction_dir: Path,
    visualization_dir: Optional[Path],
    daily_root: Path,
) -> None:
    run_date = date.today().isoformat()
    model_root = daily_root / args.model
    if args.daily_subdir:
        subdir = Path(args.daily_subdir)
        if subdir.is_absolute() or ".." in subdir.parts:
            raise SystemExit("--daily-subdir must be a relative path without '..'")
        model_root = model_root / subdir
    image_names = archive_image_names(args, visualization_dir)
    image_stems = {Path(name).stem for name in image_names}
    copied = 0

    if args.dry_run:
        print(f"Would archive daily results to: {model_root}")
        return

    if visualization_dir is not None and visualization_dir.exists():
        for image_name in image_names:
            source = visualization_dir / Path(image_name).name
            if not source.exists():
                continue
            dest = model_root / "visualizations" / prefixed_result_name(
                run_date,
                args.model,
                args.split,
                source.stem,
                source.suffix,
                "visualization",
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied += 1

    if prediction_dir.exists():
        for source in sorted(path for path in prediction_dir.rglob("*") if path.is_file()):
            if image_stems and source.stem not in image_stems:
                continue
            rel_parent = source.relative_to(prediction_dir).parent
            dest = model_root / "predictions" / rel_parent / prefixed_result_name(
                run_date,
                args.model,
                args.split,
                source.stem,
                source.suffix,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied += 1

    print(f"Daily results: {model_root} ({copied} file(s) copied)")


def open_in_vscode(visualization_dir: Optional[Path], image_names: List[str], dry_run: bool) -> None:
    if visualization_dir is None:
        return
    code_cmd = shutil.which("code")
    if code_cmd is None:
        print("VS Code command `code` was not found on PATH; visualizations were still saved.")
        return

    target = None
    if len(image_names) == 1:
        requested = visualization_dir / Path(image_names[0]).name
        if requested.exists():
            target = requested

    if target is None:
        pngs = sorted(visualization_dir.glob("*.png"))
        target = pngs[0] if len(pngs) == 1 else visualization_dir

    cmd = [code_cmd, "-r", str(target)]
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"VS Code did not open successfully. Open this path manually: {target}")


def list_models() -> None:
    width = max(len(name) for name in MODELS)
    for name, spec in sorted(MODELS.items()):
        print(f"{name:<{width}}  task={spec['task']:<16} kind={spec['kind']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--phenobench-dir", type=Path, default=os.environ.get("PHENOBENCH_DIR"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--devkit-root", type=Path, default=DEFAULT_DEVKIT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--conf-thres", type=float, default=0.0)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--viz-limit", type=int, default=16)
    parser.add_argument("--image", action="append", default=[], help="Run/visualize one specific image. Repeatable for visualization; YOLO inference supports one.")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--no-daily-output", action="store_true", help="Do not copy results into output_YYYY-MM-DD.")
    parser.add_argument("--daily-output-root", type=Path, help="Override the daily archive directory. Defaults to output_YYYY-MM-DD.")
    parser.add_argument("--daily-subdir", help="Optional subfolder under output_YYYY-MM-DD/<model> for archive copies.")
    parser.add_argument("--open-vscode", action="store_true", help="Open the visualization PNG/folder in VS Code after rendering.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        return args
    if not args.model:
        parser.error("--model is required unless --list-models is used")
    if args.phenobench_dir is None:
        parser.error("--phenobench-dir is required, or set PHENOBENCH_DIR")
    return args


def main() -> int:
    args = parse_args()
    if args.list_models:
        list_models()
        return 0

    spec = MODELS[args.model]
    output_dir = args.output_root / args.model / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Model={args.model} task={spec['task']} split={args.split} output={output_dir}")

    if "note" in spec and spec.get("kind") != "unsupported":
        print(f"Note: {spec['note']}", flush=True)

    default_weights = rel(spec["weights"]) if "weights" in spec else None
    weights = args.weights or default_weights
    if spec["kind"] not in {"external_hapt"} and weights is not None:
        weights = ensure_weight(spec, weights, args.download_weights, args.dry_run)

    if args.skip_infer:
        log_step("Skipping inference and reusing existing raw predictions")
        prediction_dir = output_dir / "predictions"
        if not args.dry_run:
            if spec["kind"] in {"semantic_lightning", "rcnn"}:
                raw_dir = output_dir / "raw"
                if raw_dir.exists():
                    log_step("Normalizing existing raw predictions")
                    normalize_known_outputs(raw_dir, prediction_dir)
            elif spec["kind"] == "weyler":
                raw_dir = output_dir / "raw"
                if raw_dir.exists():
                    log_step("Normalizing existing Weyler predictions")
                    normalize_weyler_outputs(raw_dir, prediction_dir, args.split)
            elif spec["kind"] == "docker_make":
                log_step("Normalizing existing Docker predictions")
                normalize_known_outputs(output_dir, prediction_dir)
    elif spec["kind"] == "yolov7":
        prediction_dir = run_yolov7(spec, args, output_dir, weights)
    elif spec["kind"] == "semantic_lightning":
        prediction_dir = run_semantic(spec, args, output_dir, weights)
    elif spec["kind"] == "rcnn":
        prediction_dir = run_rcnn(spec, args, output_dir, weights)
    elif spec["kind"] == "weyler":
        prediction_dir = run_weyler(spec, args, output_dir, weights)
    elif spec["kind"] == "docker_make":
        prediction_dir = run_docker_make(spec, args, output_dir, weights)
    else:
        prediction_dir = unsupported(spec, args)

    visualization_dir = visualize(spec, args, prediction_dir, output_dir)
    if not args.no_daily_output:
        daily_root = args.daily_output_root or (REPO_ROOT / f"output_{date.today().isoformat()}")
        archive_results(args, prediction_dir, visualization_dir, daily_root)
    if args.open_vscode:
        open_in_vscode(visualization_dir, args.image, args.dry_run)
    print(f"Predictions: {prediction_dir}")
    print(f"Visualizations: {output_dir / 'visualizations'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
