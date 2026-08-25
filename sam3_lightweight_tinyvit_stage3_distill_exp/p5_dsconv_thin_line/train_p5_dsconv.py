#!/usr/bin/env python3
"""训练P5-A：冻结P2全部旧参数，仅训练TinyViT Stage 2 DSConv细线分支。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import lightning as L
import torch

P5_DIR = Path(__file__).resolve().parent
EXP_DIR = P5_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STAGE3_DIR))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "p1_image_feature"))
sys.path.insert(0, str(P5_DIR))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import train_p0_image_lora as p0
from dsconv_branch import P5_BRANCH_PREFIX, attach_p5_dsconv_branch
from sam3_detr_exp.utils import CrackYoloSegDataModule
from train_p1_image_feature import P1ImageFeatureDistillModule


class P5DSConvModule(P1ImageFeatureDistillModule):
    def __init__(
        self,
        *args,
        branch_lr: float = 1e-4,
        gate_lr: float = 1e-3,
        branch_channels: int = 128,
        dsconv_kernel_size: int = 9,
        offset_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hparams.branch_lr = branch_lr
        self.hparams.gate_lr = gate_lr
        self.hparams.branch_channels = branch_channels
        self.hparams.dsconv_kernel_size = dsconv_kernel_size
        self.hparams.offset_scale = offset_scale
        self.attachment = attach_p5_dsconv_branch(
            self.detector,
            branch_channels=branch_channels,
            kernel_size=dsconv_kernel_size,
            offset_scale=offset_scale,
        )
        self._freeze_existing_parameters()
        self._set_trainable_modes()
        self._printed_shapes = False

    def _freeze_existing_parameters(self) -> None:
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        for parameter in self.detector.p5_thin_line_branch.parameters():
            parameter.requires_grad = True

    def _set_trainable_modes(self) -> None:
        # P1/P0构造期间分支尚未挂载，先保持父类初始化行为。
        if not hasattr(self, "detector") or not hasattr(
            self.detector, "p5_thin_line_branch"
        ):
            return super()._set_trainable_modes()
        # 冻结参数不等于切换推理态。SAM3根模块的training标志控制
        # Decoder辅助层、DAC和O2M输出，必须与P2保持相同训练语义。
        super()._set_trainable_modes()
        self.detector.p5_thin_line_branch.train()
        if not self.detector.training or not self.detector.transformer.training:
            raise RuntimeError("P5-A训练时SAM3根模块和Transformer必须处于train模式")
        if self.detector.backbone.training:
            raise RuntimeError("P5-A冻结Backbone必须处于eval模式")

    def configure_optimizers(self):
        gate = [self.detector.p5_thin_line_branch.gate]
        branch = [
            parameter
            for name, parameter in self.detector.p5_thin_line_branch.named_parameters()
            if name != "gate"
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": branch, "lr": self.hparams.branch_lr, "name": "dsconv"},
                {"params": gate, "lr": self.hparams.gate_lr, "name": "gate"},
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
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }

    def _shared_step(self, batch, stage: str):
        result = super()._shared_step(batch, stage)
        branch = self.detector.p5_thin_line_branch
        if not self._printed_shapes and branch.last_input_shape is not None:
            print(
                "P5形状："
                f"Stage2={branch.last_input_shape}，"
                f"融合输出={branch.last_output_shape}，"
                f"gate={float(branch.gate.detach()):.8f}",
                flush=True,
            )
            self._printed_shapes = True
        return result

    def on_after_backward(self) -> None:
        values = {
            "offset": torch.zeros((), device=self.device),
            "projection": torch.zeros((), device=self.device),
            "fusion": torch.zeros((), device=self.device),
            "gate": torch.zeros((), device=self.device),
        }
        for name, parameter in self.detector.p5_thin_line_branch.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().norm(2).square()
            if name == "gate":
                group = "gate"
            elif ".offset." in name:
                group = "offset"
            elif ".projection." in name:
                group = "projection"
            else:
                group = "fusion"
            values[group] += value
        for name, value in values.items():
            self.log(
                f"train/grad_norm_{name}", value.sqrt(),
                on_step=True, on_epoch=False,
            )

    def save_lora_checkpoint(self, path: Path) -> None:
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.detector.state_dict().items()
            if "parametrizations" in name
            or name.startswith(("dot_prod_scoring.", "segmentation_head."))
            or name.startswith(P5_BRANCH_PREFIX)
        }
        meta = {
            **self.student_source_meta,
            "experiment": "TinyViT P5-A frozen P2 plus Stage 2 DSConv branch",
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
            "p5_stage": 2,
            "p5_branch_channels": int(self.hparams.branch_channels),
            "p5_dsconv_kernel_size": int(self.hparams.dsconv_kernel_size),
            "p5_offset_scale": float(self.hparams.offset_scale),
            "p5_branch_lr": float(self.hparams.branch_lr),
            "p5_gate_lr": float(self.hparams.gate_lr),
            "p5_frozen_existing_parameters": True,
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
    parser.add_argument("--branch-lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--branch-channels", type=int, default=128)
    parser.add_argument("--dsconv-kernel-size", type=int, default=9)
    parser.add_argument("--offset-scale", type=float, default=1.0)
    parser.add_argument("--log-name", default="p5a_dsconv_frozen")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(
        student_lora=EXP_DIR / "weights/p2_image_stage123_r8.best.pt",
        save=EXP_DIR / "weights/p5a_dsconv_frozen.pt",
        epochs=2,
    )
    args = parser.parse_args()
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    p0.bind_local_cuda_device(args.accelerator)
    L.seed_everything(args.seed, workers=True)
    datamodule = CrackYoloSegDataModule(
        data_yaml=args.data_yaml,
        resolution=args.resolution,
        prompt_mode="class_name",
        generic_prompt="road marking",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    datamodule.setup("fit")
    model = P5DSConvModule(
        cache_root=args.cache_root,
        feature_cache_root=args.feature_cache_root,
        student_lora=args.student_lora,
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        image_lora_lr=args.image_lora_lr,
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
        branch_lr=args.branch_lr,
        gate_lr=args.gate_lr,
        branch_channels=args.branch_channels,
        dsconv_kernel_size=args.dsconv_kernel_size,
        offset_scale=args.offset_scale,
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    branch = sum(
        parameter.numel()
        for parameter in model.detector.p5_thin_line_branch.parameters()
    )
    non_branch_trainable = [
        name for name, parameter in model.detector.named_parameters()
        if parameter.requires_grad and not name.startswith(P5_BRANCH_PREFIX)
    ]
    if non_branch_trainable:
        raise RuntimeError(f"P5-A存在意外可训练旧参数：{non_branch_trainable[:20]}")
    print(
        f"P5-A参数：总计={total:,}，分支={branch:,}，可训练={trainable:,}，"
        f"Stage2={model.attachment.stage2_channels}x"
        f"{model.attachment.stage2_resolution}，"
        f"融合通道={model.attachment.output_channels}",
        flush=True,
    )
    best_path = args.best_save or p0.default_best_path(args.save)
    callbacks = [p0.SaveBest(best_path)]
    if args.save_every_epoch:
        callbacks.append(p0.SaveEveryEpoch(args.save))
    trainer = L.Trainer(
        default_root_dir=EXP_DIR / "logs" / args.log_name,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs,
        precision=args.precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=1 if args.dry_run else 1.0,
        limit_val_batches=1 if args.dry_run else 1.0,
        log_every_n_steps=1 if args.dry_run else 10,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存P5-A最后权重：{args.save}", flush=True)


if __name__ == "__main__":
    main()
