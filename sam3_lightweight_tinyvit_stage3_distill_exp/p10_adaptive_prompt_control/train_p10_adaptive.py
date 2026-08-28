#!/usr/bin/env python3
"""P10：从P9最佳点开始，以验证集逐类别指标闭环控制正提示频率。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping

P10_DIR = Path(__file__).resolve().parent
EXP_DIR = P10_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
P9_DIR = EXP_DIR / "p9_fresh_p8_new_teacher"
for path in (
    PROJECT_ROOT, PROJECT_ROOT / "sam3_lightweight_stage3_exp", EXP_DIR,
    EXP_DIR / "p1_image_feature", EXP_DIR / "p5_dsconv_thin_line",
    EXP_DIR / "p6_multiscale_dsconv", EXP_DIR / "p7_highres_fpn",
    EXP_DIR / "p8_input_line_branch", P9_DIR, P10_DIR,
):
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3
activate_efficientsam3()

import train_p0_image_lora as p0
from adaptive_prompt_control import (
    CLASS_NAMES, AdaptivePromptController, ControllerConfig,
    deterministic_keep, prompt_slug,
)
from sam3_detr_exp.utils import CrackYoloSegDataModule
from train_p9_fresh_p8 import P9FreshP8Module


class P10AdaptivePromptModule(P9FreshP8Module):
    def __init__(
        self, *args, prompt_seed: int = 42,
        target_dashed_solid_instance_ratio: float = 2.0,
        natural_solid_instances: int = 40017,
        natural_dashed_instances: int = 150242,
        prompt_min_rate: float = 0.4,
        prompt_ema_new_weight: float = 0.3,
        prompt_update_weight: float = 0.2,
        prompt_max_epoch_change: float = 0.1,
        prompt_deadband: float = 0.02,
        prompt_performance_gain: float = 2.0,
        prompt_min_positive_images: int = 20,
        validation_confidence_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dashed_base_rate = (
            float(target_dashed_solid_instance_ratio)
            * float(natural_solid_instances)
            / float(natural_dashed_instances)
        )
        base_rates = {class_id: 1.0 for class_id in CLASS_NAMES}
        base_rates[2] = dashed_base_rate
        config = ControllerConfig(
            minimum_rate=prompt_min_rate,
            ema_new_weight=prompt_ema_new_weight,
            update_weight=prompt_update_weight,
            maximum_epoch_change=prompt_max_epoch_change,
            deadband=prompt_deadband,
            performance_gain=prompt_performance_gain,
            minimum_positive_images=prompt_min_positive_images,
        )
        self.prompt_controller = AdaptivePromptController(base_rates, config)
        self.prompt_seed = int(prompt_seed)
        self.validation_confidence_threshold = float(validation_confidence_threshold)
        self._p10_outputs = None
        self._train_prompt_counts = None
        self._val_counts = None
        self.student_source_meta.update({
            "p10_adaptive_prompt_control": True,
            "target_dashed_solid_instance_ratio": float(
                target_dashed_solid_instance_ratio
            ),
            "natural_solid_instances": int(natural_solid_instances),
            "natural_dashed_instances": int(natural_dashed_instances),
            "white_dashed_base_prompt_rate": float(dashed_base_rate),
            "validation_confidence_threshold": self.validation_confidence_threshold,
        })

    def on_train_epoch_start(self) -> None:
        self._train_prompt_counts = torch.zeros(
            len(CLASS_NAMES), 4, dtype=torch.float64, device=self.device
        )

    def _control_train_prompts(self, batch):
        if self._train_prompt_counts is None:
            raise RuntimeError("P10训练提示计数器没有初始化")
        epoch = int(self.current_epoch)
        controlled = []
        for sample in batch:
            prompts = []
            for target in sample.prompts:
                class_id = target.class_id
                if class_id is None or target.prompt_kind != "positive":
                    prompts.append(target)
                    continue
                self._train_prompt_counts[class_id, 0] += 1
                self._train_prompt_counts[class_id, 1] += len(target.gt_boxes)
                rate = self.prompt_controller.rates[class_id]
                if deterministic_keep(
                    sample.image_path, class_id, epoch, rate, self.prompt_seed
                ):
                    prompts.append(target)
                    self._train_prompt_counts[class_id, 2] += 1
                    self._train_prompt_counts[class_id, 3] += len(target.gt_boxes)
            controlled.append(replace(sample, prompts=prompts))
        return controlled

    @contextmanager
    def _capture_grounding_outputs(self):
        original = self.detector.forward_grounding
        self._p10_outputs = None

        def wrapped(*args, **kwargs):
            outputs = original(*args, **kwargs)
            self._p10_outputs = outputs
            return outputs

        self.detector.forward_grounding = wrapped
        try:
            yield
        finally:
            self.detector.forward_grounding = original

    def _shared_step(self, batch, stage: str):
        controlled = self._control_train_prompts(batch) if stage == "train" else batch
        split = "train" if stage == "train" else "val"
        self._image_feature_kd = None
        # 图像特征KD仍使用完整标注；只有DETR监督与输出KD受提示控制。
        with self._capture_image_feature_loss(batch, split):
            with self._capture_grounding_outputs():
                total = p0.P0DistillModule._shared_step(self, controlled, stage)
        if stage == "val":
            if self._p10_outputs is None:
                raise RuntimeError("P10没有捕获到验证输出")
            self._accumulate_validation_metrics(batch, self._p10_outputs)
        return total

    def on_train_epoch_end(self) -> None:
        counts = self._train_prompt_counts
        if counts is None:
            return
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        for class_id in CLASS_NAMES:
            total_positive = counts[class_id, 0].clamp(min=1.0)
            actual_rate = counts[class_id, 2] / total_positive
            self.log(
                f"train/prompt_rate/{prompt_slug(class_id)}",
                actual_rate, sync_dist=True,
            )
            self.log(
                f"train/positive_instances_seen/{prompt_slug(class_id)}",
                counts[class_id, 3], sync_dist=True,
            )
            self.log(
                f"train/positive_instances_available/{prompt_slug(class_id)}",
                counts[class_id, 1], sync_dist=True,
            )

    def on_validation_epoch_start(self) -> None:
        # intersection、union、pred、gt、positive_images
        self._val_counts = torch.zeros(
            len(CLASS_NAMES), 5, dtype=torch.float64, device=self.device
        )

    @torch.no_grad()
    def _accumulate_validation_metrics(self, batch, outputs) -> None:
        if self._val_counts is None:
            raise RuntimeError("P10验证计数器没有初始化")
        prompt_targets = [
            target for sample in batch for target in sample.prompts
        ]
        logits = outputs["pred_logits"].detach().float().sigmoid().squeeze(-1)
        presence = (
            outputs["presence_logit_dec"].detach().float().sigmoid().reshape(-1, 1)
        )
        scores = logits * presence
        mask_logits = outputs["pred_masks"].detach().float()
        for prompt_index, target in enumerate(prompt_targets):
            class_id = target.class_id
            if class_id is None:
                continue
            if len(target.gt_masks):
                gt = target.gt_masks.to(self.device).any(dim=0)
                self._val_counts[class_id, 4] += 1
            else:
                gt = torch.zeros(
                    self.hparams.resolution, self.hparams.resolution,
                    dtype=torch.bool, device=self.device,
                )
            keep = scores[prompt_index] > self.validation_confidence_threshold
            if keep.any():
                resized = F.interpolate(
                    mask_logits[prompt_index, keep].unsqueeze(1),
                    size=gt.shape[-2:], mode="bilinear", align_corners=False,
                ).squeeze(1)
                pred = resized.sigmoid().gt(0.5).any(dim=0)
            else:
                pred = torch.zeros_like(gt)
            intersection = (pred & gt).sum(dtype=torch.float64)
            union = (pred | gt).sum(dtype=torch.float64)
            self._val_counts[class_id, 0] += intersection
            self._val_counts[class_id, 1] += union
            self._val_counts[class_id, 2] += pred.sum(dtype=torch.float64)
            self._val_counts[class_id, 3] += gt.sum(dtype=torch.float64)

    def on_validation_epoch_end(self) -> None:
        counts = self._val_counts
        if counts is None:
            return
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        iou, precision, recall, positive_images = {}, {}, {}, {}
        eligible = []
        for class_id in CLASS_NAMES:
            intersection, union, pred, gt, positives = counts[class_id]
            iou[class_id] = float((intersection / union.clamp(min=1)).cpu())
            precision[class_id] = float((intersection / pred.clamp(min=1)).cpu())
            recall[class_id] = float((intersection / gt.clamp(min=1)).cpu())
            positive_images[class_id] = int(positives.cpu())
            slug = prompt_slug(class_id)
            self.log(f"val/iou/{slug}", iou[class_id], sync_dist=True)
            self.log(f"val/precision/{slug}", precision[class_id], sync_dist=True)
            self.log(f"val/recall/{slug}", recall[class_id], sync_dist=True)
            if positive_images[class_id] >= self.prompt_controller.config.minimum_positive_images:
                eligible.append(class_id)
        if eligible:
            macro_iou = sum(iou[c] for c in eligible) / len(eligible)
            macro_recall = sum(recall[c] for c in eligible) / len(eligible)
            weakest_quality = min(
                0.7 * iou[c] + 0.3 * recall[c] for c in eligible
            )
            control_score = (
                0.5 * macro_iou + 0.3 * macro_recall + 0.2 * weakest_quality
            )
        else:
            macro_iou = macro_recall = control_score = 0.0
        self.log("val/macro_iou", macro_iou, prog_bar=True, sync_dist=True)
        self.log("val/macro_recall", macro_recall, sync_dist=True)
        self.log("val/control_score", control_score, prog_bar=True, sync_dist=True)
        next_rates = self.prompt_controller.update(
            int(self.current_epoch), iou, recall, positive_images
        )
        for class_id, rate in next_rates.items():
            self.log(
                f"control/next_prompt_rate/{prompt_slug(class_id)}",
                rate, sync_dist=True,
            )
        self.student_source_meta["prompt_controller_history"] = list(
            self.prompt_controller.history
        )
        self.student_source_meta["next_prompt_rates"] = {
            str(k): float(v) for k, v in next_rates.items()
        }
        if self.trainer.is_global_zero:
            summary = " ".join(
                f"{prompt_slug(c)}:IoU={iou[c]:.4f},R={recall[c]:.4f},"
                f"next={next_rates[c]:.3f}"
                for c in CLASS_NAMES
            )
            print(f"P10验证闭环 epoch={self.current_epoch} {summary}", flush=True)


class SaveBestHigher(L.Callback):
    def __init__(self, path: Path, monitor: str) -> None:
        self.path, self.monitor, self.best = Path(path), monitor, float("-inf")

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None or not torch.isfinite(value):
            return
        current = float(value.detach().cpu())
        if current <= self.best:
            return
        self.best = current
        if trainer.is_global_zero:
            pl_module.save_lora_checkpoint(self.path)
            print(
                f"已保存P10验证任务最佳权重：{self.path} "
                f"{self.monitor}={current:.6f}", flush=True,
            )


class SaveEveryAfterMetrics(L.Callback):
    """在P10验证指标与下一轮频率生成后保存逐轮权重。"""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        suffix = self.base_path.suffix or ".pt"
        path = self.base_path.with_name(
            f"{self.base_path.stem}.epoch{trainer.current_epoch}{suffix}"
        )
        pl_module.save_lora_checkpoint(path)
        print(f"已保存P10 epoch权重：{path}", flush=True)


def main() -> None:
    parser = p0.build_parser()
    parser.description = __doc__
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--image-feature-kd-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=4.0)
    parser.add_argument("--image-lora-stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--branch-lr", type=float, default=2e-5)
    parser.add_argument("--gate-lr", type=float, default=2e-4)
    parser.add_argument("--resume-weights", type=Path, required=True)
    parser.add_argument("--log-name", default="p10_adaptive_prompt_control")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--target-dashed-solid-instance-ratio", type=float, default=2.0)
    parser.add_argument("--prompt-min-rate", type=float, default=0.4)
    parser.add_argument("--prompt-ema-new-weight", type=float, default=0.3)
    parser.add_argument("--prompt-update-weight", type=float, default=0.2)
    parser.add_argument("--prompt-max-epoch-change", type=float, default=0.1)
    parser.add_argument("--prompt-deadband", type=float, default=0.02)
    parser.add_argument("--prompt-performance-gain", type=float, default=2.0)
    parser.add_argument("--prompt-min-positive-images", type=int, default=20)
    parser.add_argument("--validation-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.002)
    parser.set_defaults(
        data_yaml=P10_DIR / "configs/roadline_no_generic_negatives.yaml",
        cache_root=P9_DIR / "cache/new_teacher_outputs",
        save=EXP_DIR / "weights/p10_adaptive_prompt_control.pt",
        epochs=10, batch_size=4, lora_lr=1e-5,
        image_lora_lr=2e-6, head_lr=4e-6, kd_warmup_ratio=0.0,
    )
    args = parser.parse_args()
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    p0.bind_local_cuda_device(args.accelerator)
    L.seed_everything(args.seed, workers=True)
    datamodule = CrackYoloSegDataModule(
        data_yaml=args.data_yaml, resolution=args.resolution,
        prompt_mode="class_name", generic_prompt="road marking",
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples, num_generic_negatives=0,
    )
    datamodule.setup("fit")
    model = P10AdaptivePromptModule(
        cache_root=args.cache_root, feature_cache_root=args.feature_cache_root,
        checkpoint=args.checkpoint, resolution=args.resolution,
        lora_lr=args.lora_lr, head_lr=args.head_lr,
        image_lora_lr=args.image_lora_lr, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        image_lora_rank=args.image_lora_rank,
        image_lora_alpha=args.image_lora_alpha,
        image_lora_dropout=args.image_lora_dropout,
        image_lora_stages=tuple(args.image_lora_stages),
        weight_decay=args.weight_decay, kd_weight=args.kd_weight,
        kd_warmup_ratio=args.kd_warmup_ratio,
        quality_threshold=args.quality_threshold, temperature=args.temperature,
        image_feature_kd_weight=args.image_feature_kd_weight,
        foreground_weight=args.foreground_weight,
        branch_lr=args.branch_lr, gate_lr=args.gate_lr,
        resume_weights=args.resume_weights, prompt_seed=args.seed,
        target_dashed_solid_instance_ratio=args.target_dashed_solid_instance_ratio,
        prompt_min_rate=args.prompt_min_rate,
        prompt_ema_new_weight=args.prompt_ema_new_weight,
        prompt_update_weight=args.prompt_update_weight,
        prompt_max_epoch_change=args.prompt_max_epoch_change,
        prompt_deadband=args.prompt_deadband,
        prompt_performance_gain=args.prompt_performance_gain,
        prompt_min_positive_images=args.prompt_min_positive_images,
        validation_confidence_threshold=args.validation_confidence_threshold,
    )
    best_path = args.best_save or p0.default_best_path(args.save)
    callbacks = [
        SaveBestHigher(best_path, "val/control_score"),
        EarlyStopping(
            monitor="val/control_score", mode="max",
            min_delta=args.early_stop_min_delta,
            patience=args.early_stop_patience, check_finite=True, verbose=True,
        ),
    ]
    if args.save_every_epoch:
        callbacks.append(SaveEveryAfterMetrics(args.save))
    trainer = L.Trainer(
        default_root_dir=EXP_DIR / "logs" / args.log_name,
        accelerator=args.accelerator, devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs,
        precision=args.precision, gradient_clip_val=1.0,
        callbacks=callbacks, enable_checkpointing=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        limit_train_batches=1 if args.dry_run else 1.0,
        limit_val_batches=1 if args.dry_run else 1.0,
        log_every_n_steps=1 if args.dry_run else 10,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存P10最后权重：{args.save}", flush=True)


if __name__ == "__main__":
    main()
