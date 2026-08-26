from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

DEFAULT_DATA_YAML = Path("/slow_disk/ccl/data/crack_segment/data.yaml")


def image_transform(resolution: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def xyxy_to_cxcywh_normalized(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> list[float]:
    cx = ((x0 + x1) * 0.5) / width
    cy = ((y0 + y1) * 0.5) / height
    w = (x1 - x0) / width
    h = (y1 - y0) / height
    return [cx, cy, w, h]


@dataclass(frozen=True)
class YoloDatasetConfig:
    yaml_path: Path
    train_dir: Path
    val_dir: Path | None
    class_names: dict[int, str]
    prompt_training: dict


def parse_data_yaml(yaml_path: Path) -> YoloDatasetConfig:
    yaml_path = yaml_path.expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset yaml must contain a mapping: {yaml_path}")

    raw_names = raw.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        raise ValueError(f"Failed to parse class names from {yaml_path}")

    root_value = raw.get("path", ".")
    dataset_root = Path(root_value).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = yaml_path.parent / dataset_root
    dataset_root = dataset_root.resolve()

    def resolve_split(key: str, required: bool) -> Path | None:
        value = raw.get(key)
        if value in (None, ""):
            if required:
                raise ValueError(f"Missing required '{key}' entry in {yaml_path}")
            return None
        if not isinstance(value, str):
            raise ValueError(f"'{key}' must be a directory path string in {yaml_path}")
        split_path = Path(value).expanduser()
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        return split_path.resolve()

    return YoloDatasetConfig(
        yaml_path=yaml_path,
        train_dir=resolve_split("train", required=True),
        val_dir=resolve_split("val", required=False),
        class_names=names,
        prompt_training=raw.get("prompt_training", {}) or {},
    )


def polygon_to_mask(
    points_xy: list[tuple[float, float]], width: int, height: int
) -> torch.Tensor:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon(points_xy, fill=1)
    return torch.from_numpy(np.array(canvas, dtype=np.uint8)).bool()


def resize_binary_masks(masks: torch.Tensor, resolution: int) -> torch.Tensor:
    return (
        F.interpolate(
            masks.unsqueeze(1).float(),
            size=(resolution, resolution),
            mode="nearest",
        )
        .squeeze(1)
        .bool()
    )


@dataclass
class PromptTarget:
    text_prompt: str
    gt_boxes: torch.Tensor
    gt_masks: torch.Tensor
    class_id: int | None
    prompt_kind: str


@dataclass
class Sample:
    image: torch.Tensor
    prompts: list[PromptTarget]
    image_path: Path


class YoloSegmentationDataset(Dataset):
    IMAGE_SUFFIXES = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP",
    }

    def __init__(
        self,
        split_dir: Path,
        class_names: dict[int, str],
        resolution: int,
        prompt_mode: str,
        generic_prompt: str,
        prompt_training: dict | None = None,
        include_generic_negatives: bool = False,
        num_generic_negatives: int | None = None,
        max_samples: int | None = None,
    ):
        self.split_dir = split_dir
        self.resolution = resolution
        self.prompt_mode = prompt_mode
        self.generic_prompt = generic_prompt
        self.class_names = class_names
        self.prompt_training = prompt_training or {}
        self.multi_prompt = self.prompt_training.get("mode", "single_prompt") == "multi_prompt"
        self.include_generic_negatives = include_generic_negatives
        configured_num_negatives = int(self.prompt_training.get("num_negatives", 0))
        self.num_generic_negatives = (
            configured_num_negatives
            if num_generic_negatives is None
            else int(num_generic_negatives)
        )
        raw_generic_negatives = self.prompt_training.get("generic_negatives", []) or []
        if not isinstance(raw_generic_negatives, list):
            raise ValueError("prompt_training.generic_negatives must be a list")
        category_words = {
            word.lower()
            for name in self.class_names.values()
            for word in name.replace("_", " ").split()
        }
        self.generic_negatives = sorted({
            str(name).strip().lower() for name in raw_generic_negatives
            if str(name).strip()
            and not set(str(name).strip().lower().split()) & category_words
        })
        if self.num_generic_negatives < 0:
            raise ValueError("prompt_training.num_negatives must be non-negative")
        self.transform = image_transform(resolution)
        self.records = self._build_records(max_samples=max_samples)

    def _parse_label_file(self, label_path: Path) -> dict[int, list[list[float]]]:
        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for raw_line in label_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            class_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            grouped[class_id].append(coords)
        return grouped

    def _build_records(
        self, max_samples: int | None
    ) -> list[tuple[Path, dict[int, list[list[float]]]] | tuple[Path, int, list[list[float]]]]:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {self.split_dir}")

        records = []
        for image_path in sorted(self.split_dir.iterdir()):
            if image_path.suffix not in self.IMAGE_SUFFIXES:
                continue
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                continue
            grouped = self._parse_label_file(label_path)
            if self.multi_prompt:
                grouped = {
                    class_id: polygons for class_id, polygons in grouped.items()
                    if class_id in self.class_names and polygons
                }
                records.append((image_path, grouped))
                if max_samples is not None and len(records) >= max_samples:
                    return records
                continue
            for class_id, polygons in grouped.items():
                if class_id not in self.class_names or not polygons:
                    continue
                records.append((image_path, class_id, polygons))
                if max_samples is not None and len(records) >= max_samples:
                    return records

        if not records:
            raise ValueError(f"No valid YOLO segmentation samples found in {self.split_dir}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        if self.multi_prompt:
            image_path, grouped = record
        else:
            image_path, class_id, polygons = record
            grouped = {class_id: polygons}
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = self.transform(v2.functional.to_image(image))
        class_ids = sorted(self.class_names) if self.multi_prompt else list(grouped)
        prompts = []
        for class_id in class_ids:
            gt_boxes = []
            gt_masks = []
            for coords in grouped.get(class_id, []):
                xs_norm = coords[0::2]
                ys_norm = coords[1::2]
                xs = [min(max(x, 0.0), 1.0) * width for x in xs_norm]
                ys = [min(max(y, 0.0), 1.0) * height for y in ys_norm]
                if len(xs) < 3 or len(ys) < 3:
                    continue
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                if x1 <= x0 or y1 <= y0:
                    continue
                gt_boxes.append(
                    xyxy_to_cxcywh_normalized(x0, y0, x1, y1, width, height)
                )
                gt_masks.append(polygon_to_mask(list(zip(xs, ys)), width, height))
            if not self.multi_prompt and not gt_boxes:
                raise ValueError(f"No valid polygons left after parsing {image_path}")
            prompt_text = self.generic_prompt
            if self.prompt_mode == "class_name":
                prompt_text = self.class_names[class_id].replace("_", " ")
            boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4)
            masks_tensor = (
                resize_binary_masks(torch.stack(gt_masks), self.resolution)
                if gt_masks else torch.zeros(
                    0, self.resolution, self.resolution, dtype=torch.bool
                )
            )
            prompts.append(PromptTarget(
                text_prompt=prompt_text,
                gt_boxes=boxes_tensor,
                gt_masks=masks_tensor,
                class_id=class_id,
                prompt_kind="positive" if gt_boxes else "in_domain_negative",
            ))
        if self.multi_prompt and self.include_generic_negatives:
            count = min(self.num_generic_negatives, len(self.generic_negatives))
            for prompt_text in random.sample(self.generic_negatives, count):
                prompts.append(PromptTarget(
                    text_prompt=prompt_text,
                    gt_boxes=torch.zeros(0, 4, dtype=torch.float32),
                    gt_masks=torch.zeros(
                        0, self.resolution, self.resolution, dtype=torch.bool
                    ),
                    class_id=None,
                    prompt_kind="generic_negative",
                ))
        return Sample(
            image=image_tensor,
            prompts=prompts,
            image_path=image_path,
        )


def collate_samples(batch: list[Sample]) -> list[Sample]:
    return batch


class CrackYoloSegDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_yaml: Path = DEFAULT_DATA_YAML,
        resolution: int = 1008,
        prompt_mode: str = "class_name",
        generic_prompt: str = "crack",
        batch_size: int = 1,
        num_workers: int = 0,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        num_generic_negatives: int | None = None,
    ):
        super().__init__()
        self.data_yaml = data_yaml
        self.resolution = resolution
        self.prompt_mode = prompt_mode
        self.generic_prompt = generic_prompt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.num_generic_negatives = num_generic_negatives
        self.train_dataset: YoloSegmentationDataset | None = None
        self.val_dataset: YoloSegmentationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            config = parse_data_yaml(self.data_yaml)
            self.train_dataset = YoloSegmentationDataset(
                split_dir=config.train_dir,
                class_names=config.class_names,
                resolution=self.resolution,
                prompt_mode=self.prompt_mode,
                generic_prompt=self.generic_prompt,
                prompt_training=config.prompt_training,
                include_generic_negatives=True,
                num_generic_negatives=self.num_generic_negatives,
                max_samples=self.max_train_samples,
            )

            if config.val_dir is not None and config.val_dir.exists():
                self.val_dataset = YoloSegmentationDataset(
                    split_dir=config.val_dir,
                    class_names=config.class_names,
                    resolution=self.resolution,
                    prompt_mode=self.prompt_mode,
                    generic_prompt=self.generic_prompt,
                    prompt_training=config.prompt_training,
                    include_generic_negatives=False,
                    num_generic_negatives=self.num_generic_negatives,
                    max_samples=self.max_val_samples,
                )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_samples,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_samples,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
