#!/usr/bin/env python3
"""合并P2图像与DETR的r8增量，并统一转换为新初始化的r16 LoRA。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.utils.parametrize as parametrize

P3_DIR = Path(__file__).resolve().parent
EXP_DIR = P3_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STAGE3_DIR))
sys.path.insert(0, str(EXP_DIR))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

from image_lora import attach_tinyvit_image_lora
from model_adapter import build_trainable_stage3_detector
from sam3_detr_exp.utils import attach_lora_to_parametrizable_modules, save_lora_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    old_image_rank = int(meta.get("image_lora_rank", 8))
    old_image_alpha = float(meta.get("image_lora_alpha", 16.0))
    old_dropout = float(meta.get("image_lora_dropout", 0.05))
    old_detr_rank = int(meta.get("lora_rank", 8))
    old_detr_alpha = float(meta.get("lora_alpha", 16.0))
    old_detr_dropout = float(meta.get("lora_dropout", 0.05))
    model, _ = build_trainable_stage3_detector(
        checkpoint_path=args.checkpoint, text_mode="runtime", text_cache_path=None,
        lora_rank=old_detr_rank,
        lora_alpha=old_detr_alpha,
        lora_dropout=old_detr_dropout,
        decoder_only=bool(meta.get("decoder_only", False)),
        attn_only=bool(meta.get("attn_only", False)),
        train_dot_score=bool(meta.get("train_dot_score", True)),
        train_seg_head=bool(meta.get("train_seg_head", True)),
        image_lora_rank=old_image_rank, image_lora_alpha=old_image_alpha,
        image_lora_dropout=old_dropout,
        image_lora_stages=tuple(meta.get("image_lora_stages", args.stages)),
    )
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    trained_prefixes = ("dot_prod_scoring.", "segmentation_head.")
    missing = [key for key in missing if "parametrizations" in key or key.startswith(trained_prefixes)]
    unexpected = [key for key in unexpected if "parametrizations" in key or key.startswith(trained_prefixes)]
    if missing or unexpected:
        raise RuntimeError(f"P2权重不匹配：missing={missing}, unexpected={unexpected}")

    merged_image = 0
    root = model.backbone.vision_backbone
    for module in root.modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None and hasattr(parametrizations, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            module.weight.requires_grad = False
            merged_image += 1

    merged_detr = 0
    for module in model.transformer.modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name in ("weight", "in_proj_weight"):
            if not hasattr(parametrizations, parameter_name):
                continue
            parametrize.remove_parametrizations(
                module, parameter_name, leave_parametrized=True,
            )
            getattr(module, parameter_name).requires_grad = False
            merged_detr += 1

    attached_detr = attach_lora_to_parametrizable_modules(
        model=model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        include_encoder=not bool(meta.get("decoder_only", False)),
        include_decoder=True,
        include_ffn=not bool(meta.get("attn_only", False)),
    )
    attached_image = attach_tinyvit_image_lora(
        model, rank=args.rank, alpha=args.alpha,
        dropout=args.dropout, stages=tuple(args.stages),
    )
    output_meta = {
        **meta,
        "experiment": "TinyViT Stage3 all LoRA r16 initialization",
        "base_checkpoint": str(args.checkpoint),
        "student_source": str(args.input),
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "image_lora_rank": args.rank,
        "image_lora_alpha": args.alpha,
        "image_lora_dropout": args.dropout,
        "image_lora_stages": list(args.stages),
        "all_lora_conversion": {
            "merged_image_source_rank": old_image_rank,
            "merged_image_modules": merged_image,
            "new_image_modules": len(attached_image),
            "merged_detr_source_rank": old_detr_rank,
            "merged_detr_modules": merged_detr,
            "new_detr_modules": len(attached_detr),
        },
    }
    save_lora_state(model, args.output, output_meta)
    print(f"已合并图像r{old_image_rank}模块={merged_image}")
    print(f"已挂载图像r{args.rank}模块={len(attached_image)}")
    print(f"已合并DETR r{old_detr_rank}模块={merged_detr}")
    print(f"已挂载DETR r{args.rank}模块={len(attached_detr)}")
    print(f"已保存转换权重：{args.output}")


if __name__ == "__main__":
    main()
