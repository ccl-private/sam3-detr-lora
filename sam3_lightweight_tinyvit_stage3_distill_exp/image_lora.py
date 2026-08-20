from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from sam3_detr_exp.utils.detr_lora_utils import LoRAParametrization


IMAGE_LORA_PREFIX = "backbone.vision_backbone."


def attach_tinyvit_image_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    stages: tuple[int, ...] = (2, 3),
) -> list[str]:
    """给 TinyViT 高层 stage 的注意力和 MLP 线性投影挂载 LoRA。"""
    attached: list[str] = []
    stage_tokens = tuple(f".layers.{stage}.blocks." for stage in stages)
    target_suffixes = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    root = model.backbone.vision_backbone
    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(token in f".{name}" for token in stage_tokens):
            continue
        if not name.endswith(target_suffixes):
            continue
        existing = getattr(module, "parametrizations", None)
        if existing is not None and hasattr(existing, "weight"):
            continue
        parametrize.register_parametrization(
            module,
            "weight",
            LoRAParametrization(
                module.weight.shape[0], module.weight.shape[1], rank, alpha, dropout,
                device=module.weight.device, dtype=module.weight.dtype,
            ),
        )
        attached.append(f"{IMAGE_LORA_PREFIX}{name}.weight")
    if not attached:
        raise RuntimeError("没有在 TinyViT 图像编码器中找到可注入的高层线性层")
    return attached


def set_image_lora_train(model: nn.Module) -> None:
    """保持冻结骨干为 eval，只将图像 LoRA 参数化模块切到 train。"""
    for name, module in model.named_modules():
        if name.startswith(IMAGE_LORA_PREFIX) and "parametrizations" in name:
            module.train()
