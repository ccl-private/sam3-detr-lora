#!/usr/bin/env python3
"""预先提取 Stage-3 MobileCLIP-S0 文本特征，供后续缓存模式使用。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

from model_adapter import DEFAULT_STAGE3_CHECKPOINT
from sam3.model_builder import build_efficientsam3_image_model
from text_feature_provider import normalize_prompt


def prompts_from_yaml(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = config.get("names", {})
    if isinstance(names, dict):
        class_prompts = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    else:
        class_prompts = [str(value) for value in names]
    negatives = config.get("prompt_training", {}).get("generic_negatives", [])
    prompts = [normalize_prompt(prompt) for prompt in [*class_prompts, *negatives]]
    return list(dict.fromkeys(prompts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=EXP_DIR / "configs/roadline_lora.yaml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE3_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=EXP_DIR / "text_features/roadline_mobileclip_s0.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prompts = prompts_from_yaml(args.data_yaml)
    model = build_efficientsam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        device=args.device,
    ).eval()
    with torch.inference_mode():
        outputs = model.backbone.forward_text(prompts, device=torch.device(args.device))
    package = {
        "format": "efficientsam3_stage3_text_features_v1",
        "prompts": prompts,
        "attention_masks": outputs["language_mask"].detach().cpu(),
        "text_memories": outputs["language_features"].detach().cpu(),
        "text_embeds": outputs["language_embeds"].detach().cpu(),
        "metadata": {
            "source_checkpoint": str(args.checkpoint),
            "encoder": "MobileCLIP-S0",
            "context_length": 16,
            "data_yaml": str(args.data_yaml),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output)
    print(f"已保存 {len(prompts)} 个提示词的文本特征：{args.output}")


if __name__ == "__main__":
    main()
