#!/usr/bin/env python3
"""在同一批道路标线图片上比较 SAM3 Base、Stage-1 与 Stage-3。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TEST_DIR = Path(__file__).resolve().parent
EXP_DIR = TEST_DIR.parent
EFFICIENTSAM3_REPO = Path("/slow_disk/ccl/codes/efficientsam3")
sys.path.insert(0, str(EFFICIENTSAM3_REPO / "sam3"))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_efficientsam3_image_model, build_sam3_image_model

PROMPTS = {
    0: "white solid lane line",
    1: "yellow solid lane line",
    2: "white dashed lane line",
    3: "yellow dashed lane line",
    4: "zebra crossing",
    5: "lane barrier",
    6: "road teeth marking",
}


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(name: str, device: str):
    if name == "stage3":
        return build_efficientsam3_image_model(
            checkpoint_path=str(EXP_DIR / "input/efficientsam3_efficientvit_stage3.pt"),
            load_from_HF=False,
            backbone_type="efficientvit",
            model_name="b1",
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            device=device,
        ).eval()
    if name == "stage1":
        return build_efficientsam3_image_model(
            checkpoint_path=str(
                EFFICIENTSAM3_REPO / "download/efficient_sam3_efficientvit_m.pt"
            ),
            load_from_HF=False,
            backbone_type="efficientvit",
            model_name="b1",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            device=device,
        ).eval()
    if name == "base":
        return build_sam3_image_model(
            checkpoint_path=str(EXP_DIR.parent / "sam3.pt"),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            device=device,
        ).eval()
    raise ValueError(f"未知模型：{name}")


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


def render_comparison(image: Image.Image, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    canvas = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    # 真阳性为绿色、误检为红色、漏检为蓝色。
    colors = [
        (gt & pred, np.array([0, 220, 0], dtype=np.float32)),
        ((~gt) & pred, np.array([255, 40, 40], dtype=np.float32)),
        (gt & (~pred), np.array([30, 100, 255], dtype=np.float32)),
    ]
    for selected, color in colors:
        canvas[selected] = canvas[selected] * 0.35 + color * 0.65
    rendered = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    # 给每张输出图直接添加图例，避免把红色误检或蓝色漏检当成正确掩码。
    header_height = 64
    output = Image.new("RGB", (rendered.width, rendered.height + header_height), "white")
    output.paste(rendered, (0, header_height))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=22)
    legend = [
        ((0, 220, 0), "GREEN = CORRECT"),
        ((255, 40, 40), "RED = FALSE POSITIVE"),
        ((30, 100, 255), "BLUE = MISSED"),
    ]
    x = 18
    for color, label in legend:
        draw.rectangle((x, 17, x + 30, 47), fill=color)
        draw.text((x + 40, 18), label, fill="black", font=font)
        x += 290
    return output


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=TEST_DIR / "output/roadline_comparison")
    parser.add_argument("--models", nargs="+", default=["stage3", "stage1", "base"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )[: args.limit]
    if not paths:
        raise RuntimeError(f"没有找到测试图片：{args.images}")
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    model_summaries: dict[str, dict] = {}
    for model_name in args.models:
        print(f"\n===== 加载 {model_name} =====", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = load_model(model_name, args.device)
        sync()
        load_seconds = time.perf_counter() - load_start
        parameters = sum(parameter.numel() for parameter in model.parameters())
        processor = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)
        aggregate = defaultdict(int)
        elapsed_ms: list[float] = []

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            for image_index, path in enumerate(paths):
                image = Image.open(path).convert("RGB")
                width, height = image.size
                ground_truth = load_ground_truth(path.with_suffix(".txt"), width, height)
                sync()
                started = time.perf_counter()
                state = processor.set_image(image)
                sync()
                image_ms = (time.perf_counter() - started) * 1000
                for class_id, prompt in PROMPTS.items():
                    sync()
                    prompt_started = time.perf_counter()
                    result = processor.set_text_prompt(prompt, state)
                    sync()
                    prompt_ms = (time.perf_counter() - prompt_started) * 1000
                    if image_index > 0:
                        elapsed_ms.append(image_ms + prompt_ms)
                    pred = union_prediction(result["masks"], height, width)
                    gt = ground_truth[class_id]
                    intersection = int(np.count_nonzero(pred & gt))
                    union = int(np.count_nonzero(pred | gt))
                    pred_pixels = int(np.count_nonzero(pred))
                    gt_pixels = int(np.count_nonzero(gt))
                    scores = result["scores"].detach().float().cpu()
                    row = {
                        "model": model_name,
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
                        "image_encode_ms": image_ms,
                        "text_decode_ms": prompt_ms,
                    }
                    rows.append(row)
                    key = f"{model_name}|{prompt}"
                    aggregate[f"{key}|intersection"] += intersection
                    aggregate[f"{key}|union"] += union
                    aggregate[f"{key}|pred"] += pred_pixels
                    aggregate[f"{key}|gt"] += gt_pixels
                    aggregate[f"{key}|detections"] += len(scores)
                    aggregate[f"{key}|positive_gt_images"] += int(gt_pixels > 0)
                    output_dir = args.output / "visualizations" / model_name / prompt.replace(" ", "_")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    render_comparison(image, gt, pred).save(output_dir / f"{path.stem}.jpg", quality=90)
                    print(
                        f"{model_name} | {path.name} | {prompt} | "
                        f"检测={len(scores)} IoU={row['iou']:.4f}",
                        flush=True,
                    )

        prompt_summaries = {}
        for prompt in PROMPTS.values():
            key = f"{model_name}|{prompt}"
            intersection = aggregate[f"{key}|intersection"]
            union = aggregate[f"{key}|union"]
            pred_pixels = aggregate[f"{key}|pred"]
            gt_pixels = aggregate[f"{key}|gt"]
            prompt_summaries[prompt] = {
                "detections": aggregate[f"{key}|detections"],
                "positive_gt_images": aggregate[f"{key}|positive_gt_images"],
                "gt_pixels": gt_pixels,
                "micro_iou": safe_ratio(intersection, union),
                "micro_precision": safe_ratio(intersection, pred_pixels),
                "micro_recall": safe_ratio(intersection, gt_pixels),
            }
        model_summaries[model_name] = {
            "parameters": parameters,
            "load_seconds": load_seconds,
            "mean_end_to_end_ms_excluding_warmup": float(np.mean(elapsed_ms)),
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
            ),
            "prompts": prompt_summaries,
        }
        del processor, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (args.output / "details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "images": [str(path) for path in paths],
        "threshold": args.threshold,
        "visualization_legend": {"green": "true_positive", "red": "false_positive", "blue": "false_negative"},
        "models": model_summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
