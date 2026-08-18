from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


class PrecomputedTextEncoder(nn.Module):
    """与 MobileCLIP 文本编码器接口一致的预提取特征查找器。"""

    def __init__(
        self,
        prompts: Sequence[str],
        attention_masks: torch.Tensor,
        text_memories: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> None:
        super().__init__()
        normalized = tuple(normalize_prompt(prompt) for prompt in prompts)
        if len(normalized) != len(set(normalized)):
            raise ValueError("预提取文本缓存包含重复提示词")
        self.prompts = normalized
        self.prompt_to_index = {prompt: index for index, prompt in enumerate(normalized)}
        self.register_buffer("attention_masks", attention_masks, persistent=True)
        self.register_buffer("text_memories", text_memories, persistent=True)
        self.register_buffer("text_embeds", text_embeds, persistent=True)

    def forward(self, captions, input_boxes=None, device=None):
        if input_boxes is not None and input_boxes.numel() > 0:
            raise ValueError("预提取文本模式暂不支持文本中的动态框占位符")
        normalized = [normalize_prompt(caption) for caption in captions]
        unknown = sorted(set(normalized) - self.prompt_to_index.keys())
        if unknown:
            raise KeyError(
                f"文本缓存缺少提示词：{unknown}。请重新运行预提取脚本，或使用实时文本模式。"
            )
        target_device = torch.device(device) if device is not None else self.attention_masks.device
        indices = torch.tensor(
            [self.prompt_to_index[prompt] for prompt in normalized],
            dtype=torch.long,
            device=self.attention_masks.device,
        )
        masks = self.attention_masks.index_select(0, indices).to(target_device)
        memories = self.text_memories.index_select(1, indices).to(target_device)
        embeds = self.text_embeds.index_select(1, indices).to(target_device)
        return masks, memories, embeds


def load_precomputed_text_encoder(path: Path) -> tuple[PrecomputedTextEncoder, dict]:
    package = torch.load(path, map_location="cpu", weights_only=True)
    if package.get("format") != "efficientsam3_stage3_text_features_v1":
        raise ValueError(f"不支持的文本缓存格式：{package.get('format')}")
    encoder = PrecomputedTextEncoder(
        prompts=package["prompts"],
        attention_masks=package["attention_masks"],
        text_memories=package["text_memories"],
        text_embeds=package["text_embeds"],
    )
    return encoder, package.get("metadata", {})
