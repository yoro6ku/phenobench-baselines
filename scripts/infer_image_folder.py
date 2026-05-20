#!/usr/bin/env python3
"""Run a PhenoBench baseline on a plain folder of images.

The baseline wrappers expect a PhenoBench-like directory. This script creates
that structure from ordinary images, adds dummy annotation masks required by
the authors' dataloaders, then calls scripts/infer_and_visualize.py.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable, Iterator, List, Sequence, TypeVar

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVKIT_ROOT = REPO_ROOT.parent / "phenobench"
DEFAULT_PYTHON = DEFAULT_DEVKIT_ROOT / ".venv" / "bin" / "python"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
T = TypeVar("T")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "images"


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


def image_files(image_dir: Path, limit: int | None) -> List[Path]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is not None:
        images = images[:limit]
    return images


def unique_png_name(path: Path, used: set[str]) -> str:
    stem = safe_name(path.stem)
    name = f"{stem}.png"
    suffix = 1
    while name in used:
        name = f"{stem}_{suffix:03d}.png"
        suffix += 1
    used.add(name)
    return name


def assert_prepared_root_is_empty(root: Path, split: str) -> None:
    split_root = root / split
    if not split_root.exists():
        return
    existing = [path for path in split_root.rglob("*") if path.is_file()]
    if existing:
        raise SystemExit(
            f"Prepared root is not empty: {split_root}\n"
            "Choose another --prepared-root or remove the old generated files yourself."
        )


def parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if "x" in value.lower():
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    else:
        width = height = int(value)
    if width < 1 or height < 1:
        raise ValueError
    return width, height


def prepare_dataset(image_dir: Path, root: Path, split: str, limit: int | None, resize: tuple[int, int] | None) -> int:
    images = image_files(image_dir, limit)
    if not images:
        raise SystemExit(f"No supported images found in: {image_dir}")
    if resize is not None:
        print(f"Resizing prepared images to {resize[0]}x{resize[1]}", flush=True)

    split_root = root / split
    for field in ["images", "semantics", "plant_instances", "leaf_instances"]:
        (split_root / field).mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    non_square = 0
    for source in progress(images, "Preparing images", "image"):
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            if resize is not None:
                rgb = rgb.resize(resize, Image.Resampling.BILINEAR)
            width, height = rgb.size
            if width != height:
                non_square += 1

            name = unique_png_name(source, used_names)
            rgb.save(split_root / "images" / name)

            semantics = np.zeros((height, width), dtype=np.uint8)
            instances = np.zeros((height, width), dtype=np.uint16)
            Image.fromarray(semantics).save(split_root / "semantics" / name)
            Image.fromarray(instances).save(split_root / "plant_instances" / name)
            Image.fromarray(instances).save(split_root / "leaf_instances" / name)

    if non_square:
        print(
            f"Warning: {non_square} image(s) are not square. Mask outputs are fine, "
            "but author bbox txt normalization assumes square images.",
            flush=True,
        )
    print(f"Prepared {len(images)} image(s) in {split_root / 'images'}", flush=True)
    return len(images)


@contextmanager
def prepared_dataset_root(args: argparse.Namespace) -> Iterable[Path]:
    if args.prepared_root is not None:
        root = args.prepared_root.resolve()
        assert_prepared_root_is_empty(root, args.split)
        prepare_dataset(args.image_dir, root, args.split, args.limit, args.resize)
        yield root
        return

    with tempfile.TemporaryDirectory(prefix="phenobench_folder_") as temp_dir:
        root = Path(temp_dir)
        prepare_dataset(args.image_dir, root, args.split, args.limit, args.resize)
        yield root


def default_output_root(image_dir: Path) -> Path:
    run_name = f"{safe_name(image_dir.name)}_{date.today().isoformat()}"
    return REPO_ROOT / "outputs" / "image_folders" / run_name


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Folder containing images to infer.")
    parser.add_argument("--model", default="panoptic_maskrcnn", help="Wrapper model id. Default: panoptic_maskrcnn.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=Path, help="Output root. Defaults to outputs/image_folders/<folder>_<date>.")
    parser.add_argument("--prepared-root", type=Path, help="Keep the generated PhenoBench-like input folder here.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--devkit-root", type=Path, default=DEFAULT_DEVKIT_ROOT)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viz-limit", type=int, default=100000)
    parser.add_argument("--limit", type=int, help="Only prepare/infer the first N images.")
    parser.add_argument("--resize", help="Resize prepared images first, for example 1024 or 1024x1024. Defaults to 1024 for hierarchical_weyler.")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--open-vscode", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--no-daily-output", action="store_true", help="Do not also copy results into output_YYYY-MM-DD/<model>/<folder>.")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()

    args.image_dir = args.image_dir.expanduser().resolve()
    if not args.image_dir.is_dir():
        parser.error(f"image_dir is not a directory: {args.image_dir}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.viz_limit < 1:
        parser.error("--viz-limit must be >= 1")
    if args.output_root is None:
        args.output_root = default_output_root(args.image_dir)
    else:
        args.output_root = args.output_root.expanduser().resolve()
    if args.resize is None and args.model == "hierarchical_weyler":
        args.resize = (1024, 1024)
    else:
        try:
            args.resize = parse_resize(args.resize)
        except (TypeError, ValueError):
            parser.error("--resize must be SIZE or WIDTHxHEIGHT, with positive integers")
    return args, passthrough


def main() -> int:
    args, passthrough = parse_args()
    with prepared_dataset_root(args) as dataset_root:
        cmd = [
            str(args.python),
            str(REPO_ROOT / "scripts" / "infer_and_visualize.py"),
            "--model",
            args.model,
            "--phenobench-dir",
            str(dataset_root),
            "--split",
            args.split,
            "--output-root",
            str(args.output_root),
            "--python",
            str(args.python),
            "--devkit-root",
            str(args.devkit_root),
            "--device",
            args.device,
            "--viz-limit",
            str(args.viz_limit),
        ]
        if args.weights is not None:
            cmd.extend(["--weights", str(args.weights.expanduser().resolve())])
        if args.download_weights:
            cmd.append("--download-weights")
        if args.open_vscode:
            cmd.append("--open-vscode")
        if args.no_visualize:
            cmd.append("--no-visualize")
        if args.no_daily_output:
            cmd.append("--no-daily-output")
        else:
            cmd.extend(["--daily-subdir", safe_name(args.image_dir.name)])
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(passthrough)

        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    print(f"Output root: {args.output_root}", flush=True)
    print(f"Predictions: {args.output_root / args.model / args.split / 'predictions'}", flush=True)
    print(f"Visualizations: {args.output_root / args.model / args.split / 'visualizations'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
