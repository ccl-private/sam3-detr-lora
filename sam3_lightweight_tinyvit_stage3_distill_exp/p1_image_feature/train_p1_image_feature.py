#!/usr/bin/env python3
"""训练 TinyViT Stage3 P1：P0 输出 KD 加三尺度图像特征蒸馏。"""

from __future__ import annotations

import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F

P1_DIR = Path(__file__).resolve().parent
EXP_DIR = P1_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STAGE3_DIR))
sys.path.insert(0, str(EXP_DIR))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import train_p0_image_lora as p0
from sam3_detr_exp.utils import CrackYoloSegDataModule, save_lora_state


def feature_cache_file(cache_root: Path, split: str, image_path: Path) -> Path:
    identity = str(image_path.expanduser().resolve()).encode("utf-8")
    return cache_root / split / f"{hashlib.sha1(identity).hexdigest()}.pt"


class P1ImageFeatureDistillModule(p0.P0DistillModule):
    def __init__(
        self, *args, feature_cache_root: Path,
        image_feature_kd_weight: float = 1.0,
        foreground_weight: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hparams.feature_cache_root = str(feature_cache_root)
        self.hparams.image_feature_kd_weight = image_feature_kd_weight
        self.hparams.foreground_weight = foreground_weight
        self._image_feature_kd = None
        self._image_feature_metrics = {}

    def _load_teacher_features(self, batch, split: str) -> list[torch.Tensor]:
        per_image = []
        for sample in batch:
            path = feature_cache_file(
                Path(self.hparams.feature_cache_root), split, sample.image_path
            )
            if not path.exists():
                raise FileNotFoundError(f"缺少图像特征缓存：{path}，图片={sample.image_path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            per_image.append(payload["features"])
        return [
            torch.stack([features[level] for features in per_image]).to(
                self.device, non_blocking=True
            )
            for level in range(3)
        ]

    def _foreground_maps(self, batch) -> torch.Tensor:
        maps = []
        for sample in batch:
            masks = [target.gt_masks for target in sample.prompts if len(target.gt_masks)]
            if masks:
                maps.append(torch.cat(masks).any(dim=0, keepdim=True).float())
            else:
                maps.append(torch.zeros(1, self.hparams.resolution, self.hparams.resolution))
        return torch.stack(maps).to(self.device, non_blocking=True)

    def _compute_image_feature_kd(
        self, student_features, teacher_features, foreground
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = []
        metrics = {}
        for level, (student, teacher) in enumerate(zip(student_features, teacher_features)):
            student = F.adaptive_avg_pool2d(student.float(), teacher.shape[-2:])
            student = F.normalize(student, dim=1, eps=1e-6)
            teacher = F.normalize(teacher.float(), dim=1, eps=1e-6)
            pixel_loss = 1.0 - (student * teacher).sum(dim=1, keepdim=True)
            foreground_level = F.interpolate(
                foreground, size=teacher.shape[-2:], mode="nearest"
            )
            weights = 1.0 + (float(self.hparams.foreground_weight) - 1.0) * foreground_level
            loss = (pixel_loss * weights).sum() / weights.sum().clamp(min=1.0)
            losses.append(loss)
            metrics[f"kd_image_level_{level}"] = loss.detach()
        total = torch.stack(losses).mean()
        metrics["kd_image_feature"] = total.detach()
        return total, metrics

    @contextmanager
    def _capture_image_feature_loss(self, batch, split: str):
        teacher_features = self._load_teacher_features(batch, split)
        foreground = self._foreground_maps(batch)
        original = self.detector.backbone.forward_image

        def wrapped(images):
            outputs = original(images)
            self._image_feature_kd, self._image_feature_metrics = (
                self._compute_image_feature_kd(
                    outputs["backbone_fpn"], teacher_features, foreground
                )
            )
            return outputs

        self.detector.backbone.forward_image = wrapped
        try:
            yield
        finally:
            self.detector.backbone.forward_image = original

    def compute_kd(self, outputs, targets, prompt_targets, prompt_caches):
        final_kd, metrics = super().compute_kd(
            outputs, targets, prompt_targets, prompt_caches
        )
        if self._image_feature_kd is None:
            raise RuntimeError("图像特征损失没有被计算")
        metrics.update(self._image_feature_metrics)
        return (
            final_kd
            + float(self.hparams.image_feature_kd_weight) * self._image_feature_kd,
            metrics,
        )

    def _shared_step(self, batch, stage: str):
        split = "train" if stage == "train" else "val"
        self._image_feature_kd = None
        with self._capture_image_feature_loss(batch, split):
            return super()._shared_step(batch, stage)

    def save_lora_checkpoint(self, path: Path) -> None:
        meta = {
            **self.student_source_meta,
            "experiment": (
                f"TinyViT Stage3 multi-scale image feature KD, "
                f"image LoRA stages={list(self.hparams.image_lora_stages)}"
            ),
            "base_checkpoint": str(Path(self.hparams.checkpoint)),
            "student_source": str(Path(self.hparams.student_lora)),
            "teacher_cache": str(Path(self.hparams.cache_root)),
            "feature_cache_root": str(self.hparams.feature_cache_root),
            "image_feature_kd_weight": float(self.hparams.image_feature_kd_weight),
            "foreground_weight": float(self.hparams.foreground_weight),
            "kd_weight": float(self.hparams.kd_weight),
            "temperature": float(self.hparams.temperature),
            "quality_threshold": float(self.hparams.quality_threshold),
            "image_lora_rank": int(self.hparams.image_lora_rank),
            "image_lora_alpha": float(self.hparams.image_lora_alpha),
            "image_lora_dropout": float(self.hparams.image_lora_dropout),
            "image_lora_stages": list(self.hparams.image_lora_stages),
        }
        save_lora_state(self.detector, path, meta)


def main() -> None:
    parser = p0.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--feature-cache-root", type=Path,
        default=EXP_DIR / "cache/p1_image_features",
    )
    parser.add_argument("--image-feature-kd-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=4.0)
    parser.add_argument("--image-lora-stages", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--log-name", default="p1_image_feature")
    parser.set_defaults(
        student_lora=EXP_DIR / "weights/p0_image_lora_r8.best.pt",
        save=EXP_DIR / "weights/p1_image_feature_r8.pt",
    )
    args = parser.parse_args()
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    p0.bind_local_cuda_device(args.accelerator)
    L.seed_everything(args.seed, workers=True)
    datamodule = CrackYoloSegDataModule(
        data_yaml=args.data_yaml, resolution=args.resolution,
        prompt_mode="class_name", generic_prompt="road marking",
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    datamodule.setup("fit")
    model = P1ImageFeatureDistillModule(
        cache_root=args.cache_root, feature_cache_root=args.feature_cache_root,
        student_lora=args.student_lora, checkpoint=args.checkpoint,
        resolution=args.resolution, lora_lr=args.lora_lr,
        head_lr=args.head_lr, image_lora_lr=args.image_lora_lr,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        image_lora_rank=args.image_lora_rank,
        image_lora_alpha=args.image_lora_alpha,
        image_lora_dropout=args.image_lora_dropout,
        weight_decay=args.weight_decay, kd_weight=args.kd_weight,
        kd_warmup_ratio=args.kd_warmup_ratio,
        quality_threshold=args.quality_threshold, temperature=args.temperature,
        image_feature_kd_weight=args.image_feature_kd_weight,
        foreground_weight=args.foreground_weight,
        image_lora_stages=tuple(args.image_lora_stages),
    )
    print(f"P1 总参数={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"P1 可训练参数={sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}")
    best_path = args.best_save or p0.default_best_path(args.save)
    trainer = L.Trainer(
        default_root_dir=EXP_DIR / "logs" / args.log_name,
        accelerator=args.accelerator, devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs,
        precision=args.precision, gradient_clip_val=1.0,
        callbacks=[p0.SaveBest(best_path)], enable_checkpointing=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        limit_train_batches=1 if args.dry_run else 1.0,
        limit_val_batches=1 if args.dry_run else 1.0,
        log_every_n_steps=1 if args.dry_run else 10,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存 P1 最后权重：{args.save}")


if __name__ == "__main__":
    main()
