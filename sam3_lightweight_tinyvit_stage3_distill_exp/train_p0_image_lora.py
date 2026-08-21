#!/usr/bin/env python3
"""训练 TinyViT Stage3 P0：DETR LoRA + 图像编码器 LoRA + 最终层蒸馏。"""

from __future__ import annotations

import argparse
import math
import os
import sys
from functools import partial
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STAGE3_DIR))
sys.path.insert(0, str(EXP_DIR))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import sam3_detr_exp.model.detr_lora_module as module_impl
from common import load_cache, prompt_key
from image_lora import IMAGE_LORA_PREFIX, set_image_lora_train
from model_adapter import DEFAULT_STAGE3_CHECKPOINT, build_trainable_stage3_detector
from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from sam3_detr_exp.model.detr_lora_module import (
    _external_matching,
    _move_tensors_to_device,
    _reset_stale_decoder_caches,
)
from sam3_detr_exp.utils import (
    CrackYoloSegDataModule,
    build_prompt,
    build_sam3_loss_functions,
    build_targets,
    collect_trainable_parameters,
    compute_sam3_losses,
    make_find_stage,
    save_lora_state,
    set_frozen_module_modes,
)
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher


def bind_local_cuda_device(accelerator: str) -> None:
    if accelerator != "gpu":
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    print(f"CUDA 绑定：进程={os.getpid()} 本地序号={local_rank}", flush=True)


def default_best_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.best{path.suffix}")


def soft_dice_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student = student_logits.sigmoid().flatten(1)
    teacher = teacher_logits.sigmoid().flatten(1)
    numerator = 2 * (student * teacher).sum(1) + 1.0
    denominator = student.sum(1) + teacher.sum(1) + 1.0
    return 1.0 - numerator / denominator


class P0DistillModule(L.LightningModule):
    def __init__(
        self,
        cache_root: Path,
        student_lora: Path,
        checkpoint: Path,
        resolution: int = 1008,
        lora_lr: float = 5e-5,
        head_lr: float = 2e-5,
        weight_decay: float = 1e-2,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        image_lora_lr: float = 1e-5,
        image_lora_rank: int = 8,
        image_lora_alpha: float = 16.0,
        image_lora_dropout: float = 0.05,
        image_lora_stages: tuple[int, ...] = (2, 3),
        kd_weight: float = 1.0,
        kd_warmup_ratio: float = 0.05,
        quality_threshold: float = 0.2,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        detector, attached = build_trainable_stage3_detector(
            checkpoint_path=checkpoint,
            text_mode="runtime",
            text_cache_path=None,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            decoder_only=False,
            attn_only=False,
            train_dot_score=True,
            train_seg_head=True,
            image_lora_rank=image_lora_rank,
            image_lora_alpha=image_lora_alpha,
            image_lora_dropout=image_lora_dropout,
            image_lora_stages=image_lora_stages,
        )
        payload = torch.load(student_lora, map_location="cpu", weights_only=False)
        missing, unexpected = detector.load_state_dict(payload["state_dict"], strict=False)
        missing = [
            key for key in missing
            if ("parametrizations" in key and not key.startswith(IMAGE_LORA_PREFIX))
            or key.startswith(("dot_prod_scoring.", "segmentation_head."))
        ]
        unexpected = [key for key in unexpected if "parametrizations" in key or key.startswith(("dot_prod_scoring.", "segmentation_head."))]
        if missing or unexpected:
            raise RuntimeError(f"学生 LoRA 不匹配：missing={missing}, unexpected={unexpected}")
        self.detector = detector
        self.attached_lora_modules = attached
        self.student_source_meta = payload.get("meta", {})
        self.matcher = BinaryHungarianMatcherV2(
            focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
            alpha=0.25, gamma=2.0, stable=False,
        )
        self.o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
        self.sam3_loss_fns = build_sam3_loss_functions()
        set_frozen_module_modes(self.detector, train_dot_score=True, train_seg_head=True)
        set_image_lora_train(self.detector)

    def _set_trainable_modes(self) -> None:
        """设置训练态；子实验可扩展需要完整解冻的视觉模块。"""
        set_frozen_module_modes(
            self.detector, train_dot_score=True, train_seg_head=True,
        )
        set_image_lora_train(self.detector)

    def configure_optimizers(self):
        lora_params = []
        image_lora_params = []
        head_params = []
        for name, parameter in self.detector.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(("dot_prod_scoring.", "segmentation_head.")):
                head_params.append(parameter)
            elif name.startswith(IMAGE_LORA_PREFIX):
                image_lora_params.append(parameter)
            else:
                lora_params.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": self.hparams.lora_lr},
                {"params": image_lora_params, "lr": self.hparams.image_lora_lr},
                {"params": head_params, "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        def schedule(step: int) -> float:
            total = max(1, int(self.trainer.estimated_stepping_batches))
            warmup = max(1, int(total * self.hparams.kd_warmup_ratio))
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": LambdaLR(optimizer, schedule), "interval": "step"},
        }

    def kd_scale(self) -> float:
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = max(1, int(total * self.hparams.kd_warmup_ratio))
        return float(self.hparams.kd_weight) * min(1.0, (self.global_step + 1) / warmup)

    def compute_kd(self, outputs, targets, prompt_targets, prompt_caches):
        device = outputs["pred_logits"].device
        zero = outputs["pred_logits"].sum() * 0.0
        temperature = float(self.hparams.temperature)

        teacher_presence = torch.stack([
            cache["presence_logit"].float().reshape(()) for cache in prompt_caches
        ]).to(device)
        student_presence = outputs["presence_logit_dec"].float().reshape(-1)
        loss_presence = F.binary_cross_entropy_with_logits(
            student_presence / temperature,
            (teacher_presence / temperature).sigmoid(),
        ) * temperature**2

        batch_idx, src_idx, tgt_idx = self.matcher(
            outputs,
            targets,
            target_is_valid_padded=targets["target_is_valid_padded"],
        )
        offsets = []
        offset = 0
        for target in prompt_targets:
            offsets.append(offset)
            offset += len(target.gt_boxes)

        student_logits = []
        teacher_logits = []
        student_boxes = []
        teacher_boxes = []
        student_masks = []
        teacher_masks = []
        weights = []
        for position in range(len(src_idx)):
            prompt_index = int(batch_idx[position])
            local_target = int(tgt_idx[position]) - offsets[prompt_index]
            teacher_instance = prompt_caches[prompt_index]["instances"][local_target]
            if teacher_instance is None:
                continue
            quality = float(teacher_instance["quality"])
            if quality < float(self.hparams.quality_threshold):
                continue
            query_index = int(src_idx[position])
            student_logits.append(outputs["pred_logits"][prompt_index, query_index].float())
            teacher_logits.append(teacher_instance["query_logit"].float())
            student_boxes.append(outputs["pred_boxes"][prompt_index, query_index].float())
            teacher_boxes.append(teacher_instance["box"].float())
            student_masks.append(outputs["pred_masks"][prompt_index, query_index].float())
            teacher_masks.append(teacher_instance["mask_logit"].float())
            weights.append(quality)

        if not weights:
            return zero, {
                "kd_query_cls": zero.detach(), "kd_presence": loss_presence.detach(),
                "kd_box_l1": zero.detach(), "kd_box_giou": zero.detach(),
                "kd_mask_bce": zero.detach(), "kd_mask_dice": zero.detach(),
                "kd_matches": 0,
            }

        weights_t = torch.tensor(weights, device=device)
        weights_t = weights_t / weights_t.sum().clamp(min=1e-6)
        student_logits_t = torch.stack(student_logits)
        teacher_logits_t = torch.stack(teacher_logits).to(device)
        per_cls = F.binary_cross_entropy_with_logits(
            student_logits_t / temperature,
            (teacher_logits_t / temperature).sigmoid(),
            reduction="none",
        ).flatten(1).mean(1) * temperature**2
        loss_cls = (per_cls * weights_t).sum()

        student_boxes_t = torch.stack(student_boxes)
        teacher_boxes_t = torch.stack(teacher_boxes).to(device)
        loss_box_l1 = (
            F.l1_loss(student_boxes_t, teacher_boxes_t, reduction="none").mean(1) * weights_t
        ).sum()
        giou = torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(student_boxes_t), box_cxcywh_to_xyxy(teacher_boxes_t)
        ))
        loss_box_giou = ((1.0 - giou) * weights_t).sum()

        student_masks_t = torch.stack(student_masks)
        teacher_masks_t = torch.stack(teacher_masks).to(device)
        teacher_masks_t = F.interpolate(
            teacher_masks_t.unsqueeze(1), size=student_masks_t.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        per_mask_bce = F.binary_cross_entropy_with_logits(
            student_masks_t, teacher_masks_t.sigmoid(), reduction="none"
        ).flatten(1).mean(1)
        loss_mask_bce = (per_mask_bce * weights_t).sum()
        loss_mask_dice = (soft_dice_loss(student_masks_t, teacher_masks_t) * weights_t).sum()

        kd = (
            loss_cls + loss_presence + loss_box_l1 + loss_box_giou
            + 2.0 * loss_mask_bce + loss_mask_dice
        )
        return kd, {
            "kd_query_cls": loss_cls.detach(), "kd_presence": loss_presence.detach(),
            "kd_box_l1": loss_box_l1.detach(), "kd_box_giou": loss_box_giou.detach(),
            "kd_mask_bce": loss_mask_bce.detach(), "kd_mask_dice": loss_mask_dice.detach(),
            "kd_matches": len(weights),
        }

    def _shared_step(self, batch, stage: str):
        if stage == "train":
            self._set_trainable_modes()
        images = torch.stack([sample.image for sample in batch]).to(self.device, non_blocking=True)
        prompt_targets = []
        prompt_img_ids = []
        prompt_caches = []
        split = "train" if stage == "train" else "val"
        for image_index, sample in enumerate(batch):
            payload = load_cache(Path(self.hparams.cache_root), split, sample.image_path)
            for target in sample.prompts:
                key = prompt_key(target.text_prompt)
                if key not in payload["prompts"]:
                    raise KeyError(f"教师缓存缺少提示：{target.text_prompt}，图片={sample.image_path}")
                prompt_targets.append(target)
                prompt_img_ids.append(image_index)
                prompt_caches.append(payload["prompts"][key])
        texts = [target.text_prompt for target in prompt_targets]
        _reset_stale_decoder_caches(self.detector, self.device)
        backbone_out = self.detector.backbone.forward_image(images)
        with torch.no_grad():
            backbone_out.update(self.detector.backbone.forward_text(texts, device=self.device))
        backbone_out = _move_tensors_to_device(backbone_out, self.device)
        find_input = make_find_stage(torch.tensor(prompt_img_ids), self.device)
        geometric_prompt = build_prompt(self.detector, len(prompt_targets), self.device)
        with _external_matching(self.detector):
            outputs = self.detector.forward_grounding(
                backbone_out=backbone_out, find_input=find_input, find_target=None,
                geometric_prompt=geometric_prompt,
            )
        targets = build_targets(prompt_targets, self.device)
        supervised, supervised_metrics = compute_sam3_losses(
            outputs, targets, matcher=self.matcher,
            o2m_matcher=self.o2m_matcher, loss_fns=self.sam3_loss_fns,
        )
        kd, kd_metrics = self.compute_kd(outputs, targets, prompt_targets, prompt_caches)
        scale = self.kd_scale()
        total = supervised + scale * kd
        batch_size = len(batch)
        self.log(f"{stage}/loss", total, prog_bar=True, batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/supervised", supervised.detach(), batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/kd", kd.detach(), batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/kd_scale", scale, batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/num_matches", float(supervised_metrics["num_matches"]), batch_size=batch_size, sync_dist=stage == "val")
        for key, value in kd_metrics.items():
            self.log(f"{stage}/{key}", float(value) if isinstance(value, int) else value, batch_size=batch_size, sync_dist=stage == "val")
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def on_after_backward(self) -> None:
        lora_sq = torch.zeros((), device=self.device)
        image_lora_sq = torch.zeros((), device=self.device)
        head_sq = torch.zeros((), device=self.device)
        for name, parameter in self.detector.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().norm(2).square()
            if name.startswith(("dot_prod_scoring.", "segmentation_head.")):
                head_sq += value
            elif name.startswith(IMAGE_LORA_PREFIX):
                image_lora_sq += value
            else:
                lora_sq += value
        self.log("train/grad_norm_lora", lora_sq.sqrt(), on_step=True, on_epoch=False)
        self.log(
            "train/grad_norm_image_lora", image_lora_sq.sqrt(),
            on_step=True, on_epoch=False,
        )
        self.log("train/grad_norm_heads", head_sq.sqrt(), on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def save_lora_checkpoint(self, path: Path) -> None:
        meta = {
            **self.student_source_meta,
            "experiment": "TinyViT Stage3 P0-R8 + image-encoder LoRA",
            "student_source": str(Path(self.hparams.student_lora)),
            "teacher_cache": str(Path(self.hparams.cache_root)),
            "kd_weight": float(self.hparams.kd_weight),
            "temperature": float(self.hparams.temperature),
            "quality_threshold": float(self.hparams.quality_threshold),
            "image_lora_rank": int(self.hparams.image_lora_rank),
            "image_lora_alpha": float(self.hparams.image_lora_alpha),
            "image_lora_dropout": float(self.hparams.image_lora_dropout),
            "image_lora_stages": list(self.hparams.image_lora_stages),
        }
        save_lora_state(self.detector, path, meta)


class SaveBest(L.Callback):
    def __init__(self, path: Path):
        self.path = path
        self.best = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val/loss")
        if value is None or float(value) >= self.best:
            return
        self.best = float(value)
        if trainer.is_global_zero:
            pl_module.save_lora_checkpoint(self.path)
            print(f"已保存 P0 最佳权重：{self.path} val/loss={self.best:.6f}", flush=True)


class SaveEveryEpoch(L.Callback):
    """保存每个完整验证epoch，供训练后按任务IoU而不是loss选模。"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        suffix = self.base_path.suffix or ".pt"
        path = self.base_path.with_name(
            f"{self.base_path.stem}.epoch{trainer.current_epoch}{suffix}"
        )
        pl_module.save_lora_checkpoint(path)
        print(f"已保存epoch权重：{path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=STAGE3_DIR / "configs/roadline_lora.yaml")
    parser.add_argument("--cache-root", type=Path, default=EXP_DIR / "cache/p0_teacher")
    parser.add_argument("--student-lora", type=Path, default=STAGE3_DIR / "weights_lora/roadline_stage3_ev_m.best.pt")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE3_CHECKPOINT)
    parser.add_argument("--save", type=Path, default=EXP_DIR / "weights/p0_r8.pt")
    parser.add_argument("--best-save", type=Path)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--lora-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--image-lora-lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-lora-rank", type=int, default=8)
    parser.add_argument("--image-lora-alpha", type=float, default=16.0)
    parser.add_argument("--image-lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--quality-threshold", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    bind_local_cuda_device(args.accelerator)
    L.seed_everything(args.seed, workers=True)
    datamodule = CrackYoloSegDataModule(
        data_yaml=args.data_yaml, resolution=args.resolution,
        prompt_mode="class_name", generic_prompt="road marking",
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    datamodule.setup("fit")
    model = P0DistillModule(
        cache_root=args.cache_root, student_lora=args.student_lora,
        checkpoint=args.checkpoint, resolution=args.resolution,
        lora_lr=args.lora_lr, head_lr=args.head_lr,
        image_lora_lr=args.image_lora_lr,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        image_lora_rank=args.image_lora_rank,
        image_lora_alpha=args.image_lora_alpha,
        image_lora_dropout=args.image_lora_dropout,
        weight_decay=args.weight_decay, kd_weight=args.kd_weight,
        kd_warmup_ratio=args.kd_warmup_ratio,
        quality_threshold=args.quality_threshold, temperature=args.temperature,
    )
    print(f"P0 总参数={sum(p.numel() for p in model.parameters()):,}")
    print(f"P0 可训练参数={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    best_path = args.best_save or default_best_path(args.save)
    trainer = L.Trainer(
        default_root_dir=EXP_DIR / "logs/p0",
        accelerator=args.accelerator, devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs,
        precision=args.precision, gradient_clip_val=1.0,
        callbacks=[SaveBest(best_path)], enable_checkpointing=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        limit_train_batches=1 if args.dry_run else 1.0,
        limit_val_batches=1 if args.dry_run else 1.0,
        log_every_n_steps=1 if args.dry_run else 10,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存 P0 最后权重：{args.save}")


if __name__ == "__main__":
    main()
