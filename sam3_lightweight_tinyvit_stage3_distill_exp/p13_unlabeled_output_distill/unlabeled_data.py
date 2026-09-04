from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from sam3_detr_exp.utils.detr_lora_data import image_transform

DEFAULT_UNLABELED_ROOT = Path("/slow_disk/ccl/data/roadline_unlabeled_distill")


class UnlabeledRoadlineDataset(Dataset):
    """读取经过人工清理的无标签图片清单。"""

    def __init__(self, manifest: Path, resolution: int = 1008, max_samples: int | None = None):
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.exists():
            raise FileNotFoundError(f"无标签清单不存在：{self.manifest}")
        self.records = []
        for line_no, raw in enumerate(self.manifest.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            path_value = row.get("image_path")
            path = (
                Path(path_value).expanduser().resolve()
                if path_value else (DEFAULT_UNLABELED_ROOT / row["relative_path"]).resolve()
            )
            if not path.exists():
                raise FileNotFoundError(f"无标签图片不存在：{path}（清单第{line_no}行）")
            self.records.append({**row, "image_path": path})
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.transform = image_transform(resolution)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(v2.functional.to_image(image.convert("RGB")))
        return {"image": tensor, "image_path": row["image_path"]}


def collate_unlabeled(samples: list[dict]) -> dict:
    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "image_paths": [sample["image_path"] for sample in samples],
    }
