#!/usr/bin/env python3
"""缓存 Base+DETR 教师的三尺度图像特征，供 P1 图像编码器蒸馏使用。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

P1_DIR = Path(__file__).resolve().parent
EXP_DIR = P1_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam3_detr_exp.modular_pipeline import BPE_PATH, build_detector_model
from sam3_detr_exp.utils.detr_lora_data import (
    YoloSegmentationDataset,
    collate_samples,
    parse_data_yaml,
)


def feature_cache_file(cache_root: Path, split: str, image_path: Path) -> Path:
    identity = str(image_path.expanduser().resolve()).encode("utf-8")
    return cache_root / split / f"{hashlib.sha1(identity).hexdigest()}.pt"


def cache_complete(path: Path, image_path: Path, pool_factor: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return (
            payload.get("format_version") == 1
            and payload.get("image_path") == str(image_path.expanduser().resolve())
            and payload.get("pool_factor") == pool_factor
            and len(payload.get("features", [])) == 3
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=EXP_DIR / "cache/p1_image_features")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--pool-factor", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if args.pool_factor < 1:
        raise ValueError("pool-factor 必须大于等于 1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("分片参数不合法")
    config = parse_data_yaml(args.data_yaml)
    model = build_detector_model(bpe_path=str(BPE_PATH)).to(args.device).eval()
    split_dirs = {"train": config.train_dir, "val": config.val_dir}

    for split in args.splits:
        dataset = YoloSegmentationDataset(
            split_dir=split_dirs[split], class_names=config.class_names,
            resolution=args.resolution, prompt_mode="class_name",
            generic_prompt="road marking", prompt_training=config.prompt_training,
            include_generic_negatives=False, max_samples=args.max_samples,
        )
        pending = []
        for index, (image_path, _) in enumerate(dataset.records):
            if index % args.num_shards != args.shard_index:
                continue
            path = feature_cache_file(args.cache_root, split, image_path)
            if args.overwrite or not cache_complete(path, image_path, args.pool_factor):
                pending.append(index)
        loader = DataLoader(
            Subset(dataset, pending), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_samples,
            pin_memory=True, persistent_workers=args.num_workers > 0,
        )
        completed = 0
        print(f"[{split}] 分片={args.shard_index}/{args.num_shards} 待生成={len(pending)}", flush=True)
        for samples in loader:
            images = torch.stack([sample.image for sample in samples]).to(
                args.device, non_blocking=True
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
            ):
                outputs = model.backbone.forward_image(images)
                pooled = [
                    F.avg_pool2d(feature.float(), args.pool_factor, args.pool_factor)
                    .to(dtype=torch.float16).cpu()
                    for feature in outputs["backbone_fpn"]
                ]
            for batch_index, sample in enumerate(samples):
                output_path = feature_cache_file(args.cache_root, split, sample.image_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "format_version": 1,
                    "image_path": str(sample.image_path.expanduser().resolve()),
                    "split": split,
                    "resolution": args.resolution,
                    "pool_factor": args.pool_factor,
                    "features": [feature[batch_index].contiguous() for feature in pooled],
                }, output_path)
                completed += 1
            if completed % max(1, args.log_every) < len(samples):
                print(f"[{split}] 本次={completed}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
