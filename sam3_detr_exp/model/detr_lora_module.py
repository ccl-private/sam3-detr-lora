from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import lightning as L
import torch

from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3_detr_exp.utils import (
    assert_modular_weights_exist,
    build_prompt,
    build_targets,
    build_sam3_loss_functions,
    build_trainable_detector,
    collect_trainable_parameters,
    compute_losses,
    compute_sam3_losses,
    make_find_stage,
    save_lora_state,
    set_frozen_module_modes,
)


def _move_tensors_to_device(value, device: torch.device):
    """Recursively align detached backbone outputs with the current DDP rank."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    return value


def _reset_stale_decoder_caches(detector, device: torch.device) -> None:
    """Drop coordinate grids created on cuda:0 before Lightning assigns a DDP rank."""
    decoder = detector.transformer.decoder
    cache = decoder.compilable_cord_cache
    if cache is not None and any(tensor.device != device for tensor in cache):
        decoder.compilable_cord_cache = None
        decoder.compilable_stored_size = None
    if any(
        tensor.device != device
        for cached_pair in decoder.coord_cache.values()
        for tensor in cached_pair
    ):
        decoder.coord_cache.clear()


@contextmanager
def _external_matching(detector):
    """Keep SAM3 in train mode while delegating matching to this experiment."""
    original_compute_matching = detector._compute_matching
    original_back_convert = detector.back_convert
    detector._compute_matching = lambda out, targets: None
    detector.back_convert = lambda target: target
    try:
        yield
    finally:
        detector._compute_matching = original_compute_matching
        detector.back_convert = original_back_convert


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
        loss_mode: str = "simple",
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
        self.o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
        self.sam3_loss_fns = build_sam3_loss_functions() if loss_mode == "sam3" else []

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
        if stage == "train":
            set_frozen_module_modes(
                self.detector,
                train_dot_score=self.hparams.train_dot_score,
                train_seg_head=self.hparams.train_seg_head,
            )
        images = torch.stack([sample.image for sample in batch], dim=0).to(
            self.device, non_blocking=True
        )
        texts = [sample.text_prompt for sample in batch]
        _reset_stale_decoder_caches(self.detector, self.device)

        with torch.no_grad():
            backbone_out = self.detector.backbone.forward_image(images)
            backbone_out.update(self.detector.backbone.forward_text(texts, device=self.device))
            backbone_out = _move_tensors_to_device(backbone_out, self.device)

        find_input = make_find_stage(len(batch), self.device)
        geometric_prompt = build_prompt(self.detector, batch, self.device)
        with _external_matching(self.detector):
            outputs = self.detector.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
        targets = build_targets(batch, self.device)
        if self.hparams.loss_mode == "sam3":
            loss, metrics = compute_sam3_losses(
                outputs, targets, matcher=self.matcher,
                o2m_matcher=self.o2m_matcher, loss_fns=self.sam3_loss_fns,
            )
        else:
            loss, metrics = compute_losses(
                outputs, targets, matcher=self.matcher,
                resolution=self.hparams.resolution,
                mask_weight=self.hparams.mask_weight,
            )

        batch_size = len(batch)
        sync_dist = stage == "val"
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=(stage == "train"),
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        metric_keys = (
            ("loss_cls", "loss_box", "loss_giou", "loss_mask")
            if self.hparams.loss_mode == "simple"
            else ("loss_ce", "presence_loss", "loss_bbox", "loss_giou", "loss_mask", "loss_dice")
        )
        for key in metric_keys:
            if key not in metrics:
                continue
            self.log(
                f"{stage}/{key}",
                metrics[key],
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        self.log(
            f"{stage}/num_matches",
            float(metrics["num_matches"]),
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        for key in ("num_aux_outputs", "num_o2m_matches"):
            if key in metrics:
                self.log(
                    f"{stage}/{key}",
                    float(metrics[key]),
                    prog_bar=False,
                    batch_size=batch_size,
                    sync_dist=sync_dist,
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
