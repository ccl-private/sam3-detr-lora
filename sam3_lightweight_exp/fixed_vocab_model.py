from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_sam3_model,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_student_vision_backbone,
    _create_vl_backbone,
)


class FixedVocabularyTextEncoder(nn.Module):
    """Parameter-free text encoder backed by precomputed prompt tensors."""

    def __init__(
        self,
        prompts: Sequence[str],
        attention_masks: torch.Tensor,
        text_memories: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> None:
        super().__init__()
        normalized = tuple(prompt.strip().lower() for prompt in prompts)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Fixed prompt vocabulary contains duplicates")
        self.prompts = normalized
        self.prompt_to_index = {prompt: index for index, prompt in enumerate(normalized)}
        self.register_buffer("attention_masks", attention_masks, persistent=True)
        self.register_buffer("text_memories", text_memories, persistent=True)
        self.register_buffer("text_embeds", text_embeds, persistent=True)

    def forward(self, captions, input_boxes=None, device=None):
        normalized = [caption.strip().lower() for caption in captions]
        unknown = sorted(set(normalized) - self.prompt_to_index.keys())
        if unknown:
            raise ValueError(
                f"Prompts are outside the fixed vocabulary: {unknown}. "
                f"Available prompts: {list(self.prompts)}"
            )
        target_device = device or self.attention_masks.device
        indices = torch.tensor(
            [self.prompt_to_index[prompt] for prompt in normalized],
            device=self.attention_masks.device,
            dtype=torch.long,
        )
        masks = self.attention_masks.index_select(0, indices).to(target_device)
        memories = self.text_memories.index_select(1, indices).to(target_device)
        embeds = self.text_embeds.index_select(1, indices).to(target_device)
        return masks, memories, embeds


def make_fixed_vocabulary_encoder(
    prompts: Sequence[str], text_outputs: dict[str, torch.Tensor]
) -> FixedVocabularyTextEncoder:
    return FixedVocabularyTextEncoder(
        prompts=prompts,
        attention_masks=text_outputs["language_mask"].detach().cpu(),
        text_memories=text_outputs["language_features"].detach().cpu(),
        text_embeds=text_outputs["language_embeds"].detach().cpu(),
    )


def build_empty_model(fixed_text: FixedVocabularyTextEncoder):
    vision = _create_student_vision_backbone(
        backbone_type="tinyvit",
        model_name="5m",
        compile_mode=None,
        enable_inst_interactivity=False,
    )
    backbone = _create_vl_backbone(vision, fixed_text)
    return _create_sam3_model(
        backbone=backbone,
        transformer=_create_sam3_transformer(),
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=_create_segmentation_head(),
        dot_prod_scoring=_create_dot_product_scoring(),
        inst_interactive_predictor=None,
        eval_mode=True,
    )


def load_fixed_vocabulary_model(path, device="cpu"):
    package = torch.load(path, map_location="cpu", weights_only=True)
    meta = package["metadata"]
    fixed = package["fixed_text"]
    encoder = FixedVocabularyTextEncoder(
        prompts=meta["prompts"],
        attention_masks=fixed["attention_masks"],
        text_memories=fixed["text_memories"],
        text_embeds=fixed["text_embeds"],
    )
    model = build_empty_model(encoder).float()
    model.load_state_dict(package["model"], strict=True)
    return model.to(device), meta
