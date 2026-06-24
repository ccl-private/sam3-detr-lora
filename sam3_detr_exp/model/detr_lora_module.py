from __future__ import annotations

from pathlib import Path

import lightning as L
import torch

from sam3.train.matcher import BinaryHungarianMatcherV2
from sam3_detr_exp.utils import (
    assert_modular_weights_exist,
    build_prompt,
    build_targets,
    build_trainable_detector,
    collect_trainable_parameters,
    compute_losses,
    make_find_stage,
    save_lora_state,
    set_frozen_module_modes,
)


class DetrLoraLightningModule(L.LightningModule):
    def __init__(
        self,
        resolution: int = 1008,
        lr: float = 2e-4,
        weight_decay: float = 1e-2,
        mask_weight: float = 2.0,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        decoder_only: bool = False,
        attn_only: bool = False,
        train_dot_score: bool = False,
        train_seg_head: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert_modular_weights_exist()

        detector, attached = build_trainable_detector(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            decoder_only=decoder_only,
            attn_only=attn_only,
            train_dot_score=train_dot_score,
            train_seg_head=train_seg_head,
        )
        self.detector = detector
        self.attached_lora_modules = attached
        set_frozen_module_modes(
            self.detector,
            train_dot_score=train_dot_score,
            train_seg_head=train_seg_head,
        )
        self.matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2.0,
            stable=False,
        )

    def configure_optimizers(self):
        params = collect_trainable_parameters(self.detector)
        if not params:
            raise RuntimeError("No trainable parameters found after LoRA attachment.")
        return torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _shared_step(self, batch, stage: str):
        images = torch.stack([sample.image for sample in batch], dim=0).to(
            self.device, non_blocking=True
        )
        texts = [sample.text_prompt for sample in batch]

        with torch.no_grad():
            backbone_out = self.detector.backbone.forward_image(images)
            backbone_out.update(self.detector.backbone.forward_text(texts, device=self.device))

        find_input = make_find_stage(len(batch), self.device)
        geometric_prompt = build_prompt(self.detector, batch, self.device)
        outputs = self.detector.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
        targets = build_targets(batch, self.device)
        loss, metrics = compute_losses(
            outputs,
            targets,
            matcher=self.matcher,
            resolution=self.hparams.resolution,
            mask_weight=self.hparams.mask_weight,
        )

        batch_size = len(batch)
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"), batch_size=batch_size)
        for key in ("loss_cls", "loss_box", "loss_giou", "loss_mask"):
            self.log(
                f"{stage}/{key}",
                metrics[key],
                prog_bar=False,
                batch_size=batch_size,
            )
        self.log(
            f"{stage}/num_matches",
            float(metrics["num_matches"]),
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def save_lora_checkpoint(self, output_path: Path) -> None:
        meta = {
            "lora_rank": self.hparams.lora_rank,
            "lora_alpha": self.hparams.lora_alpha,
            "lora_dropout": self.hparams.lora_dropout,
            "decoder_only": self.hparams.decoder_only,
            "attn_only": self.hparams.attn_only,
            "train_dot_score": self.hparams.train_dot_score,
            "train_seg_head": self.hparams.train_seg_head,
        }
        save_lora_state(self.detector, output_path, meta)
