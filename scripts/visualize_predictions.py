#!/usr/bin/env python3
"""Visualize PhenoBench-format predictions with the PhenoBench devkit.

This script intentionally works on prediction artifacts, not model-specific
objects. Inference wrappers should normalize their outputs into folders like:

  semantics/
  plant_instances/
  leaf_instances/
  plant_bboxes/
  leaf_bboxes/

The drawing itself delegates to phenobench.visualization.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVKIT_ROOT = REPO_ROOT.parent / "phenobench"
T = TypeVar("T")


def add_devkit_to_path(devkit_root: Path) -> None:
    src = devkit_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def png_names(path: Path) -> List[str]:
    return sorted(p.name for p in path.glob("*.png"))


def progress(items: Sequence[T], desc: str, unit: str) -> Iterator[T]:
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=len(items), desc=desc, unit=unit)
        return
    except ImportError:
        pass

    total = len(items)
    print(f"{desc}: 0/{total}", flush=True)
    step = max(1, total // 10) if total else 1
    for index, item in enumerate(items, start=1):
        yield item
        if index == total or index % step == 0:
            print(f"{desc}: {index}/{total}", flush=True)


def image_names(phenobench_dir: Path, split: str, requested: Sequence[str], limit: Optional[int]) -> List[str]:
    names = png_names(phenobench_dir / split / "images")
    if requested:
        requested_set = {Path(name).name for name in requested}
        names = [name for name in names if name in requested_set]
    if limit is not None:
        names = names[:limit]
    return names


def first_existing_file(root: Path, subdirs: Iterable[str], filename: str) -> Optional[Path]:
    for subdir in subdirs:
        candidate = root / subdir / filename
        if candidate.exists():
            return candidate
    candidate = root / filename
    if candidate.exists():
        return candidate
    return None


def load_mask(root: Path, subdirs: Iterable[str], filename: str) -> Optional[np.ndarray]:
    path = first_existing_file(root, subdirs, filename)
    if path is None:
        return None
    return np.array(Image.open(path))


def load_yolo_bboxes(path: Path, width: int, height: int, task: str) -> List[Dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    values = np.genfromtxt(path, dtype=float)
    if values.size == 0:
        return []
    if values.ndim == 1:
        values = values.reshape(1, -1)

    boxes = []
    for row in values:
        if len(row) < 5:
            continue
        label = int(row[0])
        # YOLO leaf models normally emit class 0, while PhenoBench's bbox
        # drawer/evaluator use label 1 for leaves.
        if task == "leaf_detection" and label == 0:
            label = 1
        cx, cy, bw, bh = row[1:5]
        boxes.append(
            {
                "label": label,
                "center": (float(cx) * width, float(cy) * height),
                "width": float(bw) * width,
                "height": float(bh) * height,
            }
        )
    return boxes


def load_bboxes(root: Path, task: str, filename: str, image: Image.Image) -> Optional[List[Dict]]:
    stem = Path(filename).with_suffix(".txt").name
    subdirs = {
        "plant_detection": ["plant_bboxes", "predictions/plant_bboxes", "labels"],
        "leaf_detection": ["leaf_bboxes", "predictions/leaf_bboxes", "labels"],
    }[task]
    path = first_existing_file(root, subdirs, stem)
    if path is None:
        return None
    return load_yolo_bboxes(path, image.width, image.height, task)


def draw_missing(ax, image: Image.Image, title: str) -> None:
    ax.imshow(image)
    ax.text(
        0.5,
        0.5,
        "missing",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 8},
    )
    ax.set_title(title)
    ax.axis("off")


def render_prediction_panel(
    task: str,
    image: Image.Image,
    image_name: str,
    prediction_dir: Path,
    output_dir: Path,
    alpha: float,
) -> bool:
    from phenobench.visualization import draw_bboxes, draw_instances, draw_semantics

    overlays = []

    if task in {"semantics", "panoptic", "hierarchical"}:
        overlays.append(
            (
                "semantics",
                "mask",
                load_mask(
                    prediction_dir,
                    ["semantics", "predictions/semantics", "postprocess/arg_max_class", "arg_max_class"],
                    image_name,
                ),
            )
        )

    if task in {"panoptic", "hierarchical"}:
        overlays.append(
            (
                "plant instances",
                "instances",
                load_mask(
                    prediction_dir,
                    ["plant_instances", "predictions/plant_instances", "instances"],
                    image_name,
                ),
            )
        )

    if task in {"leaf_instances", "hierarchical"}:
        overlays.append(
            (
                "leaf instances",
                "instances",
                load_mask(
                    prediction_dir,
                    ["leaf_instances", "predictions/leaf_instances", "instances"],
                    image_name,
                ),
            )
        )

    if task in {"plant_detection", "leaf_detection"}:
        overlays.append((task.replace("_", " "), "bboxes", load_bboxes(prediction_dir, task, image_name, image)))

    if not overlays:
        raise ValueError(f"Unsupported task for visualization: {task}")

    ncols = 1 + len(overlays)
    fig_width = max(5.0, 4.0 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 4.5), squeeze=False)
    flat_axes = axes[0]

    flat_axes[0].imshow(image)
    flat_axes[0].set_title("image")
    flat_axes[0].axis("off")

    rendered_any_prediction = False
    for ax, (title, kind, data) in zip(flat_axes[1:], overlays):
        if data is None:
            draw_missing(ax, image, title)
            continue
        if kind == "mask":
            draw_semantics(ax, image, data, alpha=alpha)
        elif kind == "instances":
            draw_instances(ax, image, data, alpha=alpha)
        elif kind == "bboxes":
            draw_bboxes(
                ax,
                image,
                data,
                colors={
                    0: (140, 140, 140),
                    1: (0, 255, 0),
                    2: (255, 0, 0),
                    3: (0, 255, 255),
                    4: (255, 0, 255),
                },
            )
        ax.set_title(title)
        ax.axis("off")
        rendered_any_prediction = True

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / image_name, dpi=150)
    plt.close(fig)
    return rendered_any_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=["semantics", "panoptic", "leaf_instances", "plant_detection", "leaf_detection", "hierarchical"],
    )
    parser.add_argument("--phenobench-dir", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--devkit-root", type=Path, default=DEFAULT_DEVKIT_ROOT)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--image", action="append", default=[], help="Specific image filename to render. Repeatable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    add_devkit_to_path(args.devkit_root)

    names = image_names(args.phenobench_dir, args.split, args.image, args.limit)
    if not names:
        raise SystemExit("No images selected for visualization.")

    rendered = 0
    for name in progress(names, "Rendering visualizations", "image"):
        image = Image.open(args.phenobench_dir / args.split / "images" / name).convert("RGB")
        if render_prediction_panel(args.task, image, name, args.prediction_dir, args.output_dir, args.alpha):
            rendered += 1

    print(f"Rendered {len(names)} panels to {args.output_dir}")
    if rendered == 0:
        print("Warning: no prediction files were found for the selected images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
