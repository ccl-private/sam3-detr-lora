#!/usr/bin/env python3
"""P4：从P2最佳权重继续，完整解冻TinyViT Stage 3与FPN neck。"""

from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
import torch
import torch.nn.utils.parametrize as parametrize

P4_DIR = Path(__file__).resolve().parent
EXP_DIR = P4_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STAGE3_DIR))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "p1_image_feature"))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import train_p0_image_lora as p0
from sam3_detr_exp.utils import CrackYoloSegDataModule
from train_p1_image_feature import P1ImageFeatureDistillModule


STAGE3_PREFIX = "backbone.vision_backbone.trunk.model.backbone.model.layers.3."
NECK_PREFIX = "backbone.vision_backbone.convs."
HEAD_PREFIXES = ("dot_prod_scoring.", "segmentation_head.")


class P4UnfreezeStage3NeckModule(P1ImageFeatureDistillModule):
    def __init__(self, *args, stage3_lr: float = 2e-6, neck_lr: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.stage3_lr = stage3_lr
        self.hparams.neck_lr = neck_lr
        self.merged_stage3_lora = self._merge_stage3_image_lora()
        self._set_stage3_neck_requires_grad()
        self._set_trainable_modes()

    @property
    def vision_root(self):
        return self.detector.backbone.vision_backbone

    @property
    def tinyvit_stage3(self):
        return self.vision_root.trunk.model.backbone.model.layers[3]

    @property
    def fpn_neck(self):
        return self.vision_root.convs

    def _merge_stage3_image_lora(self) -> int:
        merged = 0
        for module in self.tinyvit_stage3.modules():
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is None or not hasattr(parametrizations, "weight"):
                continue
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True,
            )
            merged += 1
        if merged == 0:
            raise RuntimeError("TinyViT Stage 3没有找到可合并的图像LoRA")
        return merged

    def _set_stage3_neck_requires_grad(self) -> None:
        for parameter in self.tinyvit_stage3.parameters():
            parameter.requires_grad = True
        for parameter in self.fpn_neck.parameters():
            parameter.requires_grad = True

    def _set_trainable_modes(self) -> None:
        super()._set_trainable_modes()
        self.tinyvit_stage3.train()
        self.fpn_neck.train()

    def configure_optimizers(self):
        detr_lora = []
        image_lora = []
        stage3 = []
        neck = []
        heads = []
        for name, parameter in self.detector.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(HEAD_PREFIXES):
                heads.append(parameter)
            elif name.startswith(STAGE3_PREFIX):
                stage3.append(parameter)
            elif name.startswith(NECK_PREFIX):
                neck.append(parameter)
            elif name.startswith(p0.IMAGE_LORA_PREFIX):
                image_lora.append(parameter)
            else:
                detr_lora.append(parameter)
        groups = [
            {"params": detr_lora, "lr": self.hparams.lora_lr, "name": "detr_lora"},
            {"params": image_lora, "lr": self.hparams.image_lora_lr, "name": "image_lora"},
            {"params": stage3, "lr": self.hparams.stage3_lr, "name": "stage3_full"},
            {"params": neck, "lr": self.hparams.neck_lr, "name": "neck_full"},
            {"params": heads, "lr": self.hparams.head_lr, "name": "heads"},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._lr_schedule,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _lr_schedule(self, step: int) -> float:
        import math

        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = max(1, int(total * self.hparams.kd_warmup_ratio))
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def on_after_backward(self) -> None:
        groups = {
            "detr_lora": torch.zeros((), device=self.device),
            "image_lora": torch.zeros((), device=self.device),
            "stage3": torch.zeros((), device=self.device),
            "neck": torch.zeros((), device=self.device),
            "heads": torch.zeros((), device=self.device),
        }
        for name, parameter in self.detector.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().norm(2).square()
            if name.startswith(HEAD_PREFIXES):
                groups["heads"] += value
            elif name.startswith(STAGE3_PREFIX):
                groups["stage3"] += value
            elif name.startswith(NECK_PREFIX):
                groups["neck"] += value
            elif name.startswith(p0.IMAGE_LORA_PREFIX):
                groups["image_lora"] += value
            else:
                groups["detr_lora"] += value
        for name, value in groups.items():
            self.log(f"train/grad_norm_{name}", value.sqrt(), on_step=True, on_epoch=False)

    def save_lora_checkpoint(self, path: Path) -> None:
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.detector.state_dict().items()
            if "parametrizations" in name
            or name.startswith(HEAD_PREFIXES)
            or name.startswith(STAGE3_PREFIX)
            or name.startswith(NECK_PREFIX)
        }
        meta = {
            **self.student_source_meta,
            "experiment": "TinyViT Stage3 P4 full Stage 3 and neck fine-tuning",
            "base_checkpoint": str(Path(self.hparams.checkpoint)),
            "student_source": str(Path(self.hparams.student_lora)),
            "teacher_cache": str(Path(self.hparams.cache_root)),
            "feature_cache_root": str(self.hparams.feature_cache_root),
            "image_feature_kd_weight": float(self.hparams.image_feature_kd_weight),
            "foreground_weight": float(self.hparams.foreground_weight),
            "kd_weight": float(self.hparams.kd_weight),
            "temperature": float(self.hparams.temperature),
            "quality_threshold": float(self.hparams.quality_threshold),
            "lora_rank": int(self.hparams.lora_rank),
            "lora_alpha": float(self.hparams.lora_alpha),
            "lora_dropout": float(self.hparams.lora_dropout),
            "image_lora_rank": int(self.hparams.image_lora_rank),
            "image_lora_alpha": float(self.hparams.image_lora_alpha),
            "image_lora_dropout": float(self.hparams.image_lora_dropout),
            "image_lora_stages": [1, 2],
            "unfrozen_vision_stage": 3,
            "unfrozen_neck": True,
            "merged_stage3_lora_modules": int(self.merged_stage3_lora),
            "stage3_lr": float(self.hparams.stage3_lr),
            "neck_lr": float(self.hparams.neck_lr),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": meta, "state_dict": state}, path)


def main() -> None:
    parser = p0.build_parser()
    parser.description = __doc__
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--image-feature-kd-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=4.0)
    parser.add_argument("--image-lora-stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--stage3-lr", type=float, default=2e-6)
    parser.add_argument("--neck-lr", type=float, default=1e-5)
    parser.add_argument("--log-name", default="p4_unfreeze_stage3_neck")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(
        student_lora=EXP_DIR / "weights/p2_image_stage123_r8.best.pt",
        save=EXP_DIR / "weights/p4_unfreeze_stage3_neck.pt",
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
        max_val_samples=args.max_val_samples,
    )
    datamodule.setup("fit")
    model = P4UnfreezeStage3NeckModule(
        cache_root=args.cache_root,
        feature_cache_root=args.feature_cache_root,
        student_lora=args.student_lora,
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        image_lora_lr=args.image_lora_lr,
        stage3_lr=args.stage3_lr,
        neck_lr=args.neck_lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        image_lora_rank=args.image_lora_rank,
        image_lora_alpha=args.image_lora_alpha,
        image_lora_dropout=args.image_lora_dropout,
        image_lora_stages=tuple(args.image_lora_stages),
        weight_decay=args.weight_decay,
        kd_weight=args.kd_weight,
        kd_warmup_ratio=args.kd_warmup_ratio,
        quality_threshold=args.quality_threshold,
        temperature=args.temperature,
        image_feature_kd_weight=args.image_feature_kd_weight,
        foreground_weight=args.foreground_weight,
    )
    counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stage3": sum(p.numel() for n, p in model.detector.named_parameters() if p.requires_grad and n.startswith(STAGE3_PREFIX)),
        "neck": sum(p.numel() for n, p in model.detector.named_parameters() if p.requires_grad and n.startswith(NECK_PREFIX)),
    }
    print(f"P4参数统计={counts}，已合并Stage3图像LoRA={model.merged_stage3_lora}")
    best_path = args.best_save or p0.default_best_path(args.save)
    callbacks = [p0.SaveBest(best_path)]
    if args.save_every_epoch:
        callbacks.append(p0.SaveEveryEpoch(args.save))
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
        print(f"已保存P4最后权重：{args.save}")


if __name__ == "__main__":
    main()
