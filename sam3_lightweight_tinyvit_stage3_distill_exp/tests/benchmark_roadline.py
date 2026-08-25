#!/usr/bin/env python3
"""在同一批道路标线图片上评测 TinyViT 蒸馏 LoRA，并与 Base+DETR 对比。"""

from __future__ import annotations

import argparse
import csv
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
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STAGE3_DIR))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "p5_dsconv_thin_line"))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

from sam3.model.sam3_image_processor import Sam3Processor
from dsconv_branch import P5_BRANCH_PREFIX, attach_p5_dsconv_branch
from model_adapter import DEFAULT_STAGE3_CHECKPOINT, build_trainable_stage3_detector

PROMPTS = {
    0: "white solid lane line",
    1: "yellow solid lane line",
    2: "white dashed lane line",
    3: "yellow dashed lane line",
    4: "zebra crossing",
    5: "lane barrier",
    6: "road teeth marking",
}


def load_ground_truth(path: Path, width: int, height: int) -> dict[int, np.ndarray]:
    masks = {class_id: Image.new("1", (width, height), 0) for class_id in PROMPTS}
    drawers = {class_id: ImageDraw.Draw(mask) for class_id, mask in masks.items()}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 7 or int(fields[0]) not in PROMPTS:
                continue
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            points = [(values[i] * width, values[i + 1] * height) for i in range(0, len(values) - 1, 2)]
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


def render(image: Image.Image, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    canvas = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for selected, color in (
        (gt & pred, np.array([0, 220, 0])),
        ((~gt) & pred, np.array([255, 40, 40])),
        (gt & (~pred), np.array([30, 100, 255])),
    ):
        canvas[selected] = canvas[selected] * 0.35 + color * 0.65
    body = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    output = Image.new("RGB", (body.width, body.height + 64), "white")
    output.paste(body, (0, 64))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=22)
    for x, color, label in (
        (18, (0, 220, 0), "GREEN = CORRECT"),
        (308, (255, 40, 40), "RED = FALSE POSITIVE"),
        (598, (30, 100, 255), "BLUE = MISSED"),
    ):
        draw.rectangle((x, 17, x + 30, 47), fill=color)
        draw.text((x + 40, 18), label, fill="black", font=font)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE3_CHECKPOINT)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
    if not paths:
        raise RuntimeError(f"没有找到测试图片：{args.images}")
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    model, attached = build_trainable_stage3_detector(
        checkpoint_path=args.checkpoint, text_mode="runtime", text_cache_path=None,
        lora_rank=int(meta["lora_rank"]), lora_alpha=float(meta["lora_alpha"]),
        lora_dropout=float(meta["lora_dropout"]), decoder_only=bool(meta["decoder_only"]),
        attn_only=bool(meta["attn_only"]), train_dot_score=bool(meta["train_dot_score"]),
        train_seg_head=bool(meta["train_seg_head"]), image_lora_rank=int(meta["image_lora_rank"]),
        image_lora_alpha=float(meta["image_lora_alpha"]), image_lora_dropout=float(meta["image_lora_dropout"]),
        image_lora_stages=tuple(meta["image_lora_stages"]),
    )
    model_label = "tinyvit_p0_image_lora"
    if bool(meta.get("p8_input_line_branch", False)):
        p8_dir = EXP_DIR / "p8_input_line_branch"
        p7_dir = EXP_DIR / "p7_highres_fpn"
        p6_dir = EXP_DIR / "p6_multiscale_dsconv"
        for path in (p8_dir, p7_dir, p6_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from input_line_branch import attach_p8_from_checkpoint

        attach_p8_from_checkpoint(model, args.weights)
        model_label = f"tinyvit_p8_input_{meta.get('p8_operator', 'dsconv')}"
    elif bool(meta.get("p7_highres_fpn", False)):
        p7_dir = EXP_DIR / "p7_highres_fpn"
        p6_dir = EXP_DIR / "p6_multiscale_dsconv"
        for path in (p7_dir, p6_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from highres_fpn import attach_p7_from_checkpoint

        attach_p7_from_checkpoint(model, args.weights)
        model_label = "tinyvit_p7_highres_fpn"
    elif bool(meta.get("p6_multiscale_dsconv", False)):
        p6_dir = EXP_DIR / "p6_multiscale_dsconv"
        if str(p6_dir) not in sys.path:
            sys.path.insert(0, str(p6_dir))
        from multiscale_dsconv import attach_multiscale_from_checkpoint

        attach_multiscale_from_checkpoint(model, args.weights)
        model_label = "tinyvit_p6_multiscale_dsconv"
    elif int(meta.get("p5_stage", -1)) == 2:
        attach_p5_dsconv_branch(
            model,
            branch_channels=int(meta.get("p5_branch_channels", 128)),
            kernel_size=int(meta.get("p5_dsconv_kernel_size", 9)),
            offset_scale=float(meta.get("p5_offset_scale", 1.0)),
        )
        model_label = "tinyvit_p5a_dsconv"
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    trained_prefixes = (
        "dot_prod_scoring.", "segmentation_head.", P5_BRANCH_PREFIX,
        "p6_stage1_thin_line_branch.",
        "p7_highres_fpn_adapters.",
        "p8_input_line_branch.",
    )
    missing = [key for key in missing if "parametrizations" in key or key.startswith(trained_prefixes)]
    unexpected = [key for key in unexpected if "parametrizations" in key or key.startswith(trained_prefixes)]
    if missing or unexpected:
        raise RuntimeError(f"权重不匹配：missing={missing}, unexpected={unexpected}")
    model = model.to(args.device).eval()
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, aggregate, elapsed = [], defaultdict(int), []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for image_index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            width, height = image.size
            ground_truth = load_ground_truth(path.with_suffix(".txt"), width, height)
            torch.cuda.synchronize(); started = time.perf_counter()
            state = processor.set_image(image)
            for class_id, prompt in PROMPTS.items():
                result = processor.set_text_prompt(prompt, state)
                torch.cuda.synchronize()
                if image_index > 0:
                    elapsed.append((time.perf_counter() - started) * 1000)
                pred, gt = union_prediction(result["masks"], height, width), ground_truth[class_id]
                intersection, union = int(np.count_nonzero(pred & gt)), int(np.count_nonzero(pred | gt))
                pred_pixels, gt_pixels = int(np.count_nonzero(pred)), int(np.count_nonzero(gt))
                scores = result["scores"].detach().float().cpu()
                row = {"model": model_label, "image": path.name, "prompt": prompt,
                       "threshold": args.threshold, "detections": len(scores),
                       "mean_score": float(scores.mean()) if len(scores) else 0.0,
                       "max_score": float(scores.max()) if len(scores) else 0.0,
                       "intersection_pixels": intersection, "union_pixels": union,
                       "pred_pixels": pred_pixels, "gt_pixels": gt_pixels,
                       "iou": safe_ratio(intersection, union),
                       "precision": safe_ratio(intersection, pred_pixels),
                       "recall": safe_ratio(intersection, gt_pixels)}
                rows.append(row)
                for suffix, value in (("intersection", intersection), ("union", union), ("pred", pred_pixels),
                                      ("gt", gt_pixels), ("detections", len(scores))):
                    aggregate[f"{prompt}|{suffix}"] += value
                out = args.output / "visualizations" / model_label / prompt.replace(" ", "_")
                out.mkdir(parents=True, exist_ok=True)
                render(image, gt, pred).save(out / f"{path.stem}.jpg", quality=90)
                print(f"{path.name} | {prompt} | 检测={len(scores)} IoU={row['iou']:.4f}", flush=True)

    prompts = {}
    for prompt in PROMPTS.values():
        prompts[prompt] = {
            "detections": aggregate[f"{prompt}|detections"], "gt_pixels": aggregate[f"{prompt}|gt"],
            "micro_iou": safe_ratio(aggregate[f"{prompt}|intersection"], aggregate[f"{prompt}|union"]),
            "micro_precision": safe_ratio(aggregate[f"{prompt}|intersection"], aggregate[f"{prompt}|pred"]),
            "micro_recall": safe_ratio(aggregate[f"{prompt}|intersection"], aggregate[f"{prompt}|gt"]),
        }
    base = json.loads(args.base_summary.read_text(encoding="utf-8"))
    report = {"images": [str(p) for p in paths], "threshold": args.threshold,
              "weights": str(args.weights), "attached_lora_modules": len(attached),
              "visualization_legend": {"green": "正确", "red": "误检", "blue": "漏检"},
              "models": {model_label: {"parameters": sum(p.numel() for p in model.parameters()),
                  "mean_elapsed_ms_excluding_warmup": float(np.mean(elapsed)),
                  "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20, "prompts": prompts},
                  "base_detr_lora": {"prompts": base["prompts"], "source_summary": str(args.base_summary)}}}
    with (args.output / "details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
