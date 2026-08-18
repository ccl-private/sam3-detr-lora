from __future__ import annotations

from pathlib import Path

from sam3_detr_exp.utils.detr_lora_utils import (
    attach_lora_to_parametrizable_modules,
    freeze_module,
)

from sam3_lightweight_exp.fixed_vocab_model import load_fixed_vocabulary_model


DEFAULT_PRETRAINED = (
    Path(__file__).resolve().parent
    / "weights_pretrained/roadline_tinyvit_s_fixed_vocab_fp16.pt"
)


def build_trainable_lightweight_detector(
    pretrained_path: Path,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    decoder_only: bool,
    attn_only: bool,
    train_dot_score: bool,
    train_seg_head: bool,
):
    model, metadata = load_fixed_vocabulary_model(pretrained_path, device="cpu")
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
    model.lightweight_metadata = metadata
    return model, attached
