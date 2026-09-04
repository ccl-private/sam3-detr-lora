#!/usr/bin/env python3
"""在全部无标签网图上比较P12与P13-A，并保存无真值语义的并排可视化。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

TEST_DIR = Path(__file__).resolve().parent
P13_DIR = TEST_DIR.parent
EXP_DIR = P13_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
for path in (
    PROJECT_ROOT, STAGE3_DIR, EXP_DIR, EXP_DIR / "p5_dsconv_thin_line",
    EXP_DIR / "p6_multiscale_dsconv", EXP_DIR / "p7_highres_fpn",
    EXP_DIR / "p8_input_line_branch", P13_DIR,
):
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3
activate_efficientsam3()

from input_line_branch import attach_p8_from_checkpoint
from model_adapter import build_trainable_stage3_detector
from sam3.model.sam3_image_processor import Sam3Processor


PROMPTS = [
    "white solid lane line", "yellow solid lane line", "white dashed lane line",
    "yellow dashed lane line", "zebra crossing", "lane barrier", "road teeth marking",
]
COLORS = np.asarray([
    [0, 230, 255], [255, 210, 0], [0, 130, 255], [255, 120, 0],
    [210, 50, 255], [255, 40, 90], [40, 230, 100],
], dtype=np.float32)


def read_relative_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["relative_path"] for line in path.read_text().splitlines() if line.strip()}


def load_model(weights: Path, checkpoint: Path, device: torch.device):
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    model, _ = build_trainable_stage3_detector(
        checkpoint_path=checkpoint, text_mode="runtime", text_cache_path=None,
        lora_rank=int(meta["lora_rank"]), lora_alpha=float(meta["lora_alpha"]),
        lora_dropout=float(meta["lora_dropout"]), decoder_only=bool(meta["decoder_only"]),
        attn_only=bool(meta["attn_only"]), train_dot_score=bool(meta["train_dot_score"]),
        train_seg_head=bool(meta["train_seg_head"]), image_lora_rank=int(meta["image_lora_rank"]),
        image_lora_alpha=float(meta["image_lora_alpha"]),
        image_lora_dropout=float(meta["image_lora_dropout"]),
        image_lora_stages=tuple(meta["image_lora_stages"]),
    )
    attach_p8_from_checkpoint(model, weights)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    trained = ("dot_prod_scoring.", "segmentation_head.", "p5_", "p6_", "p7_", "p8_")
    missing = [key for key in missing if "parametrizations" in key or key.startswith(trained)]
    unexpected = [key for key in unexpected if "parametrizations" in key or key.startswith(trained)]
    if missing or unexpected:
        raise RuntimeError(f"权重不匹配：{weights}，missing={missing}，unexpected={unexpected}")
    return model.to(device).eval()


def union_mask(value: torch.Tensor, height: int, width: int) -> np.ndarray:
    if len(value) == 0:
        return np.zeros((height, width), dtype=bool)
    masks = value.detach().float()
    if masks.ndim == 4:
        masks = masks[:, 0]
    if tuple(masks.shape[-2:]) != (height, width):
        masks = F.interpolate(masks.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
    return (masks > 0.5).any(0).cpu().numpy()


def infer(processor: Sam3Processor, image: Image.Image) -> tuple[list[np.ndarray], list[dict]]:
    width, height = image.size
    state = processor.set_image(image)
    masks, stats = [], []
    for prompt in PROMPTS:
        result = processor.set_text_prompt(prompt, state)
        mask = union_mask(result["masks"], height, width)
        scores = result["scores"].detach().float().cpu()
        masks.append(mask)
        stats.append({
            "prompt": prompt,
            "detections": int(len(scores)),
            "max_score": float(scores.max()) if len(scores) else 0.0,
            "pred_pixels": int(mask.sum()),
        })
    return masks, stats


def fit_panel(image: Image.Image, width: int = 520, height: int = 420) -> Image.Image:
    body = image.copy()
    body.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (25, 25, 25))
    canvas.paste(body, ((width - body.width) // 2, (height - body.height) // 2))
    return canvas


def overlay(image: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, color in zip(masks, COLORS):
        array[mask] = 0.45 * array[mask] + 0.55 * color
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def difference(image: Image.Image, p12: list[np.ndarray], p13: list[np.ndarray]) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    a = np.logical_or.reduce(p12)
    b = np.logical_or.reduce(p13)
    for mask, color in ((a & ~b, [40, 130, 255]), (a & b, [40, 220, 80]), (~a & b, [255, 140, 20])):
        array[mask] = 0.35 * array[mask] + 0.65 * np.asarray(color)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def render(image: Image.Image, p12: list[np.ndarray], p13: list[np.ndarray], relative: str, split: str) -> Image.Image:
    panels = [fit_panel(value) for value in (image, overlay(image, p12), overlay(image, p13), difference(image, p12, p13))]
    top, bottom = 58, 62
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height + top + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for index, (panel, title) in enumerate(zip(panels, ("ORIGINAL", "P12", "P13-A", "DIFF"))):
        x = index * panel.width
        canvas.paste(panel, (x, top))
        draw.text((x + 12, 10), title, fill="black", font=font)
    draw.text((12, 34), f"split={split}  {relative}", fill=(50, 50, 50), font=ImageFont.load_default(size=13))
    draw.text((12, top + panels[0].height + 8), "DIFF: blue=P12 only, green=overlap, orange=P13-A only; colors are not correctness labels", fill="black", font=ImageFont.load_default(size=14))
    legend = " | ".join(f"{i}:{name}" for i, name in enumerate(PROMPTS))
    draw.text((12, top + panels[0].height + 33), legend, fill=(60, 60, 60), font=ImageFont.load_default(size=11))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--p12-weights", type=Path, required=True)
    parser.add_argument("--p13-weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()]
    rows = sorted((row for row in rows if row.get("status") == "accepted"), key=lambda row: row["relative_path"])
    if args.limit is not None:
        rows = rows[:args.limit]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    train_set, eval_set = read_relative_paths(args.train_manifest), read_relative_paths(args.eval_manifest)
    device = torch.device(args.device)
    p12_model = load_model(args.p12_weights, args.checkpoint, device)
    p13_model = load_model(args.p13_weights, args.checkpoint, device)
    p12_processor = Sam3Processor(p12_model, device=args.device, confidence_threshold=args.threshold)
    p13_processor = Sam3Processor(p13_model, device=args.device, confidence_threshold=args.threshold)
    part_path = args.output / "parts" / f"part_{args.shard_index:02d}.jsonl"
    part_path.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for index, row in enumerate(rows, 1):
            relative = row["relative_path"]
            path = args.data_root / relative
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            p12_masks, p12_stats = infer(p12_processor, image)
            p13_masks, p13_stats = infer(p13_processor, image)
            split = "train_candidate" if relative in train_set else "heldout_eval" if relative in eval_set else "unselected"
            output_path = args.output / "visualizations" / split / Path(relative).with_suffix(".jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render(image, p12_masks, p13_masks, relative, split).save(output_path, quality=88)
            prompt_rows = []
            for prompt_index, prompt in enumerate(PROMPTS):
                a, b = p12_masks[prompt_index], p13_masks[prompt_index]
                intersection, union = int((a & b).sum()), int((a | b).sum())
                prompt_rows.append({
                    "prompt": prompt, "p12": p12_stats[prompt_index], "p13a": p13_stats[prompt_index],
                    "agreement_iou": intersection / union if union else 1.0,
                    "p12_only_pixels": int((a & ~b).sum()), "p13a_only_pixels": int((~a & b).sum()),
                })
            completed.append({
                "relative_path": relative, "split": split, "category": row.get("candidate", {}).get("category"),
                "platform": row.get("candidate", {}).get("platform"), "visualization": str(output_path),
                "prompts": prompt_rows,
            })
            if index % 10 == 0 or index == len(rows):
                print(f"分片{args.shard_index}：{index}/{len(rows)}", flush=True)
    part_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completed))


if __name__ == "__main__":
    main()
