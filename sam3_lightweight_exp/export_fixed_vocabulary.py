#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import torch
import yaml

EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam3_lightweight_exp.bootstrap import DEFAULT_SOURCE_CHECKPOINT, activate_efficientsam3

activate_efficientsam3()

from sam3.model_builder import build_efficientsam3_image_model
from sam3_lightweight_exp.fixed_vocab_model import make_fixed_vocabulary_encoder


def read_prompts(yaml_path: Path) -> list[str]:
    config = yaml.safe_load(yaml_path.read_text())
    names = config["names"]
    if isinstance(names, dict):
        class_prompts = [names[key] for key in sorted(names, key=lambda value: int(value))]
    else:
        class_prompts = names
    generic = config.get("prompt_training", {}).get("generic_negatives", []) or []
    prompts = []
    for prompt in [*class_prompts, *generic]:
        normalized = str(prompt).replace("_", " ").strip().lower()
        if normalized and normalized not in prompts:
            prompts.append(normalized)
    return prompts


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument(
        "--output", type=Path,
        default=EXP_ROOT / "weights_pretrained/roadline_tinyvit_s_fixed_vocab_fp16.pt",
    )
    args = parser.parse_args()

    prompts = read_prompts(args.data_yaml)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_efficientsam3_image_model(
        checkpoint_path=str(args.source_checkpoint),
        backbone_type="tinyvit",
        model_name="5m",
        enable_segmentation=True,
        enable_inst_interactivity=False,
        device=device,
        eval_mode=True,
    )
    with torch.inference_mode():
        text_outputs = model.backbone.forward_text(prompts, device=device)
    fixed_text = make_fixed_vocabulary_encoder(prompts, text_outputs)
    model.backbone.language_backbone = fixed_text.to(device)
    state = {
        key: value.half().cpu() if value.is_floating_point() else value.cpu()
        for key, value in model.state_dict().items()
    }
    package = {
        "metadata": {
            "format": "efficientsam3-fixed-vocabulary-v1",
            "source_checkpoint": str(args.source_checkpoint),
            "backbone_type": "tinyvit",
            "model_name": "5m",
            "prompts": prompts,
            "dtype": "fp16",
        },
        "fixed_text": {
            "attention_masks": fixed_text.attention_masks.cpu(),
            "text_memories": fixed_text.text_memories.cpu(),
            "text_embeds": fixed_text.text_embeds.cpu(),
        },
        "model": state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output)
    floating = sum(value.numel() for value in state.values() if value.is_floating_point())
    print(f"prompts={prompts}")
    print(f"floating_values={floating:,}")
    print(f"saved={args.output} size_mib={args.output.stat().st_size / 2**20:.2f}")


if __name__ == "__main__":
    main()
