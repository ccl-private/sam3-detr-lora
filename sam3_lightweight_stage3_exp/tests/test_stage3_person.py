#!/usr/bin/env python3
"""Test the official Stage-3 EV-M model with the text prompt ``person``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_efficientsam3_image_model
from sam3.visualization_utils import plot_results


TEST_DIR = Path(__file__).resolve().parent
EXP_DIR = TEST_DIR.parent


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=EXP_DIR / "input/efficientsam3_efficientvit_stage3.pt"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=TEST_DIR / "output/person_stage3.png")
    parser.add_argument("--metrics", default=TEST_DIR / "output/person_stage3_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        device=args.device,
    ).eval()
    sync()
    load_seconds = time.perf_counter() - start

    processor = Sam3Processor(
        model, device=args.device, confidence_threshold=args.threshold
    )
    image = Image.open(args.image).convert("RGB")

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
    ):
        # Warm up once, then report a second complete image+text pass.
        state = processor.set_image(image)
        state = processor.set_text_prompt("person", state)
        sync()
        start = time.perf_counter()
        state = processor.set_image(image)
        sync()
        encoded = time.perf_counter()
        state = processor.set_text_prompt("person", state)
        sync()
        prompted = time.perf_counter()

    scores = state["scores"].float().cpu().tolist()
    metrics = {
        "model": "EfficientSAM3 Stage-3 EV-M",
        "vision_encoder": "EfficientViT-B1",
        "text_encoder": "MobileCLIP-S0 ctx16",
        "prompt": "person",
        "threshold": args.threshold,
        "detections": len(scores),
        "scores": scores,
        "load_seconds": load_seconds,
        "image_encode_ms": (encoded - start) * 1000,
        "text_and_decode_ms": (prompted - encoded) * 1000,
        "end_to_end_ms": (prompted - start) * 1000,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        ),
    }
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    plot_results(image, state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle("EfficientSAM3 Stage-3 EV-M | prompt: person")
    plt.savefig(args.output, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
