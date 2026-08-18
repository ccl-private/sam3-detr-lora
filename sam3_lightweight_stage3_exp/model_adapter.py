from __future__ import annotations

from pathlib import Path

from sam3.model_builder import build_efficientsam3_image_model
from sam3_detr_exp.utils.detr_lora_utils import (
    attach_lora_to_parametrizable_modules,
    freeze_module,
)

from text_feature_provider import load_precomputed_text_encoder


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_STAGE3_CHECKPOINT = EXP_DIR / "input/efficientsam3_efficientvit_stage3.pt"


def build_trainable_stage3_detector(
    checkpoint_path: Path,
    text_mode: str,
    text_cache_path: Path | None,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    decoder_only: bool,
    attn_only: bool,
    train_dot_score: bool,
    train_seg_head: bool,
):
    model = build_efficientsam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        device="cpu",
    )
    text_metadata = {
        "mode": text_mode,
        "encoder": "MobileCLIP-S0",
        "context_length": 16,
    }
    if text_mode == "precomputed":
        if text_cache_path is None:
            raise ValueError("预提取文本模式必须提供 text_cache_path")
        cached_encoder, cached_metadata = load_precomputed_text_encoder(text_cache_path)
        model.backbone.language_backbone = cached_encoder
        text_metadata.update({"cache": str(text_cache_path), **cached_metadata})
    elif text_mode != "runtime":
        raise ValueError(f"未知文本特征模式：{text_mode}")

    freeze_module(model)
    if train_dot_score:
        for parameter in model.dot_prod_scoring.parameters():
            parameter.requires_grad = True
    if train_seg_head:
        for parameter in model.segmentation_head.parameters():
            parameter.requires_grad = True
    attached = attach_lora_to_parametrizable_modules(
        model=model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        include_encoder=not decoder_only,
        include_decoder=True,
        include_ffn=not attn_only,
    )
    model.stage3_lora_metadata = {
        "base_checkpoint": str(checkpoint_path),
        "text_features": text_metadata,
    }
    return model, attached
