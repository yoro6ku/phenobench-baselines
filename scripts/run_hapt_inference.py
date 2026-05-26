#!/usr/bin/env python3
"""Run external PRBonn/HAPT code and export PhenoBench-format predictions."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Dict, Iterator, List, Optional, Sequence, TypeVar

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAPT_ROOT = REPO_ROOT.parent / "HAPT"
DEFAULT_CONFIG = REPO_ROOT / "hiearchical_panoptic_segmentation" / "HAPT" / "PhenoBenchConfig.yaml"
DEFAULT_DATASET_ADAPTER = REPO_ROOT / "hiearchical_panoptic_segmentation" / "HAPT" / "PhenoBenchDataset.py"
DEFAULT_WEIGHTS = REPO_ROOT / "hiearchical_panoptic_segmentation" / "HAPT" / "weights" / "hapt_model.ckpt"
T = TypeVar("T")


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


def requested_images(phenobench_dir: Path, split: str, requested: List[str]) -> List[str]:
    image_dir = phenobench_dir / split / "images"
    if requested:
        names = [Path(name).name for name in requested]
    else:
        names = [path.name for path in sorted(image_dir.glob("*.png"))]
    missing = [name for name in names if not (image_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing image(s) in {image_dir}: {', '.join(missing)}")
    return names


def requested_indices(image_list: Sequence[str], requested: List[str]) -> List[int]:
    if not requested:
        return list(range(len(image_list)))
    name_to_index = {name: index for index, name in enumerate(image_list)}
    missing = [name for name in requested if name not in name_to_index]
    if missing:
        raise SystemExit(f"Missing image(s) in HAPT dataset adapter: {', '.join(missing)}")
    return [name_to_index[name] for name in requested]


def patch_cpu_cuda_calls() -> None:
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    nn.Module.cuda = lambda self, *args, **kwargs: self


def install_import_shims() -> None:
    # The public HAPT repo imports a ResNet module that is not shipped there.
    # The PhenoBench checkpoint uses the ERFNet encoder, so this path is unused.
    resnet = types.ModuleType("models.resnet")

    class _UnusedResNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Unused HAPT ResNet encoder stub was instantiated")

    resnet.ResNet18 = _UnusedResNet
    resnet.ResNet34 = _UnusedResNet
    resnet.NonBottleneck1D = object
    sys.modules["models.resnet"] = resnet

    import torchmetrics

    class _InferenceOnlyIoU:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            device = args[0].device if args else "cpu"
            return torch.zeros(2, device=device)

    torchmetrics.IoU = _InferenceOnlyIoU


def load_config(config_path: Path, phenobench_dir: Path, batch_size: int) -> dict:
    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    cfg["data"]["ft-path"] = str(phenobench_dir)
    cfg["train"]["workers"] = 0
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["n_gpus"] = 1
    return cfg


def torch_load_checkpoint(weights: Path, device: torch.device) -> dict:
    try:
        return torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(weights, map_location=device)


def load_hapt_model(hapt_root: Path, config_path: Path, weights: Path, phenobench_dir: Path, device: torch.device):
    if not (hapt_root / "models" / "HAPT.py").exists():
        raise SystemExit(
            f"HAPT code not found at {hapt_root}.\n"
            "Clone it with: git clone https://github.com/PRBonn/HAPT.git /home/hhadhri/Bureau/code/HAPT"
        )
    sys.path.insert(0, str(hapt_root))
    if device.type == "cpu":
        patch_cpu_cuda_calls()
    install_import_shims()

    from models.HAPT import Hapt

    cfg = load_config(config_path, phenobench_dir, batch_size=1)
    model = Hapt(cfg)
    checkpoint = torch_load_checkpoint(weights, device)
    checkpoint_state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    checkpoint_sem_loss = checkpoint_state.get("sem_loss.weight")
    if checkpoint_sem_loss is not None and tuple(model.sem_loss.weight.shape) != tuple(checkpoint_sem_loss.shape):
        from models.loss import mIoULoss

        # The public HAPT.py still builds a 2-weight loss, while the released
        # PhenoBench checkpoint stores the 3-class loss buffer. The loss is not
        # used for inference, but matching it lets us load the author checkpoint
        # strictly instead of silently dropping a tensor.
        model.sem_loss = mIoULoss(checkpoint_sem_loss.detach().cpu().tolist())

    model.load_state_dict(checkpoint_state, strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded HAPT checkpoint strictly: {len(checkpoint_state)} tensors", flush=True)
    return model


def load_hapt_dataset(adapter_path: Path, hapt_root: Path, phenobench_dir: Path, split: str):
    if str(hapt_root) not in sys.path:
        sys.path.insert(0, str(hapt_root))
    spec = importlib.util.spec_from_file_location("phenobench_hapt_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load HAPT PhenoBench adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PhenoRobPlantsBase(phenobench_dir / split)


def parse_size(value: str) -> Optional[tuple[int, int]]:
    if value.lower() in {"native", "none"}:
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


def save_png(array: torch.Tensor, path: Path, dtype: np.dtype, output_size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array.detach().cpu().numpy().astype(dtype))
    if image.size != output_size:
        image = image.resize(output_size, Image.Resampling.NEAREST)
    image.save(path)


def resize_chw_tensor(image: torch.Tensor, input_size: Optional[tuple[int, int]]) -> torch.Tensor:
    if input_size is None:
        return image
    width, height = input_size
    resized = torch.nn.functional.interpolate(
        image.unsqueeze(0).float(),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    return resized.squeeze(0)


def run_one_tensor(
    model,
    image: torch.Tensor,
    image_name: str,
    output_dir: Path,
    device: torch.device,
    input_size: Optional[tuple[int, int]],
) -> None:

    from utils.post_processing import our_instance

    with torch.no_grad():
        original_size = (int(image.shape[2]), int(image.shape[1]))
        image = resize_chw_tensor(image, input_size).unsqueeze(0).float().to(device)
        sem_logits, plant_centers, plant_offsets, leaf_centers, leaf_offsets = model(image)

        semantics = torch.argmax(torch.softmax(sem_logits[0], dim=0), dim=0).to(torch.uint8)

        max_plant_center = torch.max(plant_centers[0])
        max_leaf_center = torch.max(leaf_centers[0])

        plant_instances = our_instance(
            semantics.unsqueeze(0),
            plant_centers[0].unsqueeze(0),
            plant_offsets[0].unsqueeze(0),
            threshold=0.85 * max_plant_center,
            nms_kernel=41,
            grouping_dist=20.0,
        ).to(torch.uint16)

        leaf_instances = our_instance(
            semantics.unsqueeze(0),
            leaf_centers[0].unsqueeze(0),
            leaf_offsets[0].unsqueeze(0),
            threshold=0.75 * max_leaf_center,
            nms_kernel=11,
            grouping_dist=2.0,
        ).to(torch.uint16)

    save_png(semantics, output_dir / "semantics" / image_name, np.uint8, original_size)
    save_png(plant_instances, output_dir / "plant_instances" / image_name, np.uint16, original_size)
    save_png(leaf_instances, output_dir / "leaf_instances" / image_name, np.uint16, original_size)


def run_one_image(
    model,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    input_size: Optional[tuple[int, int]],
) -> None:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    run_one_tensor(
        model,
        tensor,
        image_path.name,
        output_dir,
        device,
        input_size,
    )


def run_one_sample(
    model,
    sample: Dict[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
    input_size: Optional[tuple[int, int]],
) -> None:
    image = sample["image"] if "image" in sample else sample["images"]
    image_name = sample["image_name"]
    run_one_tensor(
        model,
        image,
        image_name,
        output_dir,
        device,
        input_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hapt-root", type=Path, default=DEFAULT_HAPT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-adapter", type=Path, default=DEFAULT_DATASET_ADAPTER)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--phenobench-dir", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--input-size",
        default="native",
        help="HAPT inference size as WIDTHxHEIGHT. Use 'native' to keep original image size.",
    )
    parser.add_argument(
        "--data-source",
        default="adapter",
        choices=["adapter", "image"],
        help="Use the released HAPT PhenoBench adapter or load images directly.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"HAPT bridge: data_source={args.data_source}, author post-processing", flush=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable to this Python process; falling back to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    try:
        input_size = parse_size(args.input_size)
    except ValueError:
        raise SystemExit("--input-size must be WIDTHxHEIGHT, SIZE, or native")

    names = requested_images(args.phenobench_dir, args.split, args.image)
    size_text = "native" if input_size is None else f"{input_size[0]}x{input_size[1]}"
    print(f"HAPT inference on {len(names)} image(s), device={device}, input_size={size_text}", flush=True)
    if input_size is not None and input_size[0] != input_size[1]:
        print("Warning: non-square HAPT input size will distort square PhenoBench images.", flush=True)
    model = load_hapt_model(args.hapt_root, args.config, args.weights, args.phenobench_dir, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.data_source == "adapter":
        dataset = load_hapt_dataset(args.dataset_adapter, args.hapt_root, args.phenobench_dir, args.split)
        for index in progress(requested_indices(dataset.image_list, names), "HAPT inference", "image"):
            run_one_sample(
                model,
                dataset[index],
                args.output_dir,
                device,
                input_size,
            )
    else:
        for image_name in progress(names, "HAPT inference", "image"):
            run_one_image(
                model,
                args.phenobench_dir / args.split / "images" / image_name,
                args.output_dir,
                device,
                input_size,
            )

    print(f"Predictions: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
