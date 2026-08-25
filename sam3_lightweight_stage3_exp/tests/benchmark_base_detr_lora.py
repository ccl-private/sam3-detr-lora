#!/usr/bin/env python3
"""用与轻量 Stage-3 相同的真值口径评测 Base DETR LoRA。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3_detr_exp.modular_pipeline import BPE_PATH, build_detector_model
from sam3_detr_exp.utils import assert_modular_weights_exist, load_lora_state

PROMPTS = {
    0: "white solid lane line",
    1: "yellow solid lane line",
    2: "white dashed lane line",
    3: "yellow dashed lane line",
    4: "zebra crossing",
    5: "lane barrier",
    6: "road teeth marking",
}


def load_ground_truth(label_path: Path, width: int, height: int) -> dict[int, np.ndarray]:
    masks = {class_id: Image.new("1", (width, height), 0) for class_id in PROMPTS}
    drawers = {class_id: ImageDraw.Draw(mask) for class_id, mask in masks.items()}
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 7:
                continue
            class_id = int(fields[0])
            if class_id not in PROMPTS:
                continue
            values = [float(value) for value in fields[1:]]
            points = [
                (values[index] * width, values[index + 1] * height)
                for index in range(0, len(values) - 1, 2)
            ]
            if len(points) >= 3:
                drawers[class_id].polygon(points, fill=1)
    return {class_id: np.asarray(mask, dtype=bool) for class_id, mask in masks.items()}


def union_prediction(masks: torch.Tensor, height: int, width: int) -> np.ndarray:
    if len(masks) == 0:
        return np.zeros((height, width), dtype=bool)
    array = masks.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[:, 0]
    return np.any(array > 0.5, axis=0)


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    global PROMPTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prompt", help="只测试一个任意文本提示；该提示没有道路标线真值")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.prompt:
        PROMPTS = {-1: args.prompt}

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    assert_modular_weights_exist()
    paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )[: args.limit]
    if not paths:
        raise RuntimeError(f"没有找到测试图片：{args.images}")
    args.output.mkdir(parents=True, exist_ok=True)

    model = build_detector_model(bpe_path=str(BPE_PATH))
    meta, missing, unexpected = load_lora_state(model, args.lora)
    if missing or unexpected:
        raise RuntimeError(f"LoRA 权重不匹配：missing={missing}, unexpected={unexpected}")
    model = model.to(args.device).eval()
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)
    rows: list[dict] = []
    aggregate = defaultdict(int)

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
    ):
        for path in paths:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            ground_truth = load_ground_truth(path.with_suffix(".txt"), width, height)
            state = processor.set_image(image)
            for class_id, prompt in PROMPTS.items():
                result = processor.set_text_prompt(prompt, state)
                pred = union_prediction(result["masks"], height, width)
                gt = ground_truth[class_id]
                intersection = int(np.count_nonzero(pred & gt))
                union = int(np.count_nonzero(pred | gt))
                pred_pixels = int(np.count_nonzero(pred))
                gt_pixels = int(np.count_nonzero(gt))
                scores = result["scores"].detach().float().cpu()
                rows.append(
                    {
                        "model": "base_detr_lora",
                        "image": path.name,
                        "prompt": prompt,
                        "threshold": args.threshold,
                        "detections": len(scores),
                        "mean_score": float(scores.mean()) if len(scores) else 0.0,
                        "max_score": float(scores.max()) if len(scores) else 0.0,
                        "intersection_pixels": intersection,
                        "union_pixels": union,
                        "pred_pixels": pred_pixels,
                        "gt_pixels": gt_pixels,
                        "iou": safe_ratio(intersection, union),
                        "precision": safe_ratio(intersection, pred_pixels),
                        "recall": safe_ratio(intersection, gt_pixels),
                    }
                )
                key = prompt
                aggregate[f"{key}|intersection"] += intersection
                aggregate[f"{key}|union"] += union
                aggregate[f"{key}|pred"] += pred_pixels
                aggregate[f"{key}|gt"] += gt_pixels
                aggregate[f"{key}|detections"] += len(scores)
                print(f"{path.name} | {prompt} | 检测={len(scores)} IoU={rows[-1]['iou']:.4f}", flush=True)

    summaries = {}
    for prompt in PROMPTS.values():
        intersection = aggregate[f"{prompt}|intersection"]
        union = aggregate[f"{prompt}|union"]
        pred_pixels = aggregate[f"{prompt}|pred"]
        gt_pixels = aggregate[f"{prompt}|gt"]
        summaries[prompt] = {
            "detections": aggregate[f"{prompt}|detections"],
            "gt_pixels": gt_pixels,
            "micro_iou": safe_ratio(intersection, union),
            "micro_precision": safe_ratio(intersection, pred_pixels),
            "micro_recall": safe_ratio(intersection, gt_pixels),
        }
    with (args.output / "details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "images": [str(path) for path in paths],
        "threshold": args.threshold,
        "lora": str(args.lora),
        "lora_meta": meta,
        "prompts": summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
