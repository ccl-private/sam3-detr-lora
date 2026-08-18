#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

import cv2
from PIL import Image
import torch

EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam3_lightweight_exp.bootstrap import activate_efficientsam3

activate_efficientsam3()

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS
from sam3_detr_exp.utils import load_lora_state
from sam3_lightweight_exp.fixed_vocab_model import load_fixed_vocabulary_model
from sam3_lightweight_exp.model_adapter import DEFAULT_PRETRAINED


def render(image, boxes, masks, scores, title):
    import numpy as np
    canvas = np.array(image.convert("RGB"), copy=True)
    for index, mask in enumerate(masks):
        color = (COLORS[index % len(COLORS)] * 255).astype(np.uint8)
        mask_array = mask.detach().cpu().numpy()
        if mask_array.ndim == 3:
            mask_array = mask_array[0]
        selected = mask_array > 0.5
        canvas[selected] = (0.45 * color + 0.55 * canvas[selected]).astype(np.uint8)
    for index, box in enumerate(boxes):
        color = tuple(int(value) for value in COLORS[index % len(COLORS)] * 255)
        x0, y0, x1, y1 = [int(round(value)) for value in box.detach().cpu().tolist()]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, f"{float(scores[index]):.3f}", (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    header = np.full((40, canvas.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return np.concatenate([header, canvas], axis=0)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EXP_ROOT / "outputs/first10")
    parser.add_argument("--prompts", nargs="+", default=["white solid lane line", "white dashed lane line"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model, metadata = load_fixed_vocabulary_model(args.pretrained, device="cuda")
    if args.lora:
        meta, missing, unexpected = load_lora_state(model, args.lora)
        if missing or unexpected:
            raise RuntimeError(f"LoRA mismatch: missing={missing}, unexpected={unexpected}")
        print(f"loaded LoRA: {args.lora} meta={meta}")
    model.eval()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})[:10]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        processor = Sam3Processor(model, device="cuda", confidence_threshold=args.threshold)
        for path in paths:
            image = Image.open(path).convert("RGB")
            state = processor.set_image(image)
            for prompt in args.prompts:
                result = processor.set_text_prompt(prompt, state)
                scores = result["scores"].detach().float().cpu()
                out_dir = args.output / prompt.replace(" ", "_")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{path.stem}_vis.png"
                overlay = render(image, result["boxes"], result["masks"], result["scores"], prompt)
                cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                rows.append({"image": path.name, "prompt": prompt, "detections": len(scores), "mean_score": scores.mean().item() if len(scores) else 0.0, "max_score": scores.max().item() if len(scores) else 0.0, "output": str(out_path)})
                print(f"{path.name} | {prompt} | detections={len(scores)}")
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"model prompts={metadata['prompts']}")


if __name__ == "__main__":
    main()
