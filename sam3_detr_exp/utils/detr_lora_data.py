from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
class Sample:
    image: torch.Tensor
    text_prompt: str
    prompt_box_cxcywh: list[float] | None
    gt_boxes: torch.Tensor
    gt_masks: torch.Tensor
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
        max_samples: int | None = None,
    ):
        self.split_dir = split_dir
        self.resolution = resolution
        self.prompt_mode = prompt_mode
        self.generic_prompt = generic_prompt
        self.class_names = class_names
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
    ) -> list[tuple[Path, int, list[list[float]]]]:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {self.split_dir}")

        records: list[tuple[Path, int, list[list[float]]]] = []
        for image_path in sorted(self.split_dir.iterdir()):
            if image_path.suffix not in self.IMAGE_SUFFIXES:
                continue
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                continue
            grouped = self._parse_label_file(label_path)
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
        image_path, class_id, polygons = self.records[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        gt_boxes = []
        gt_masks = []
        for coords in polygons:
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

        if not gt_boxes:
            raise ValueError(f"No valid polygons left after parsing {image_path}")

        prompt_text = self.generic_prompt
        if self.prompt_mode == "class_name":
            prompt_text = self.class_names[class_id].replace("_", " ")

        image_tensor = self.transform(v2.functional.to_image(image))
        return Sample(
            image=image_tensor,
            text_prompt=prompt_text,
            prompt_box_cxcywh=None,
            gt_boxes=torch.tensor(gt_boxes, dtype=torch.float32),
            gt_masks=resize_binary_masks(torch.stack(gt_masks, dim=0), self.resolution),
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
                max_samples=self.max_train_samples,
            )

            if config.val_dir is not None and config.val_dir.exists():
                self.val_dataset = YoloSegmentationDataset(
                    split_dir=config.val_dir,
                    class_names=config.class_names,
                    resolution=self.resolution,
                    prompt_mode=self.prompt_mode,
                    generic_prompt=self.generic_prompt,
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
