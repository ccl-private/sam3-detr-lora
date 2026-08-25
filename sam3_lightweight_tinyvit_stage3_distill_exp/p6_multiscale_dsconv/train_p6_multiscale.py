#!/usr/bin/env python3
"""训练P6：冻结P5最佳模型，只训练新增TinyViT Stage 1 DSConv分支。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import lightning as L
import torch

P6_DIR = Path(__file__).resolve().parent
EXP_DIR = P6_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
P5_DIR = EXP_DIR / "p5_dsconv_thin_line"
for path in (PROJECT_ROOT, STAGE3_DIR, EXP_DIR, EXP_DIR / "p1_image_feature", P5_DIR, P6_DIR):
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import train_p0_image_lora as p0
from dsconv_branch import P5_BRANCH_PREFIX
from multiscale_dsconv import P6_BRANCH_PREFIX, restore_p5_then_attach_p6
from sam3_detr_exp.utils import CrackYoloSegDataModule
from train_p1_image_feature import P1ImageFeatureDistillModule


class P6MultiscaleDSConvModule(P1ImageFeatureDistillModule):
    def __init__(
        self,
        *args,
        p5_checkpoint: Path,
        stage1_branch_lr: float = 1e-4,
        stage1_gate_lr: float = 1e-3,
        stage1_branch_channels: int = 64,
        stage1_kernel_size: int = 9,
        stage1_offset_scale: float = 1.0,
        **kwargs,
    ) -> None:
        # 父类先从同一个P5文件恢复LoRA和输出头；P5分支随后显式挂载并恢复。
        super().__init__(*args, student_lora=p5_checkpoint, **kwargs)
        self.hparams.p5_checkpoint = str(p5_checkpoint)
        self.hparams.stage1_branch_lr = stage1_branch_lr
        self.hparams.stage1_gate_lr = stage1_gate_lr
        self.hparams.stage1_branch_channels = stage1_branch_channels
        self.hparams.stage1_kernel_size = stage1_kernel_size
        self.hparams.stage1_offset_scale = stage1_offset_scale
        self.p5_meta, self.attachment = restore_p5_then_attach_p6(
            self.detector,
            checkpoint_path=p5_checkpoint,
            stage1_branch_channels=stage1_branch_channels,
            stage1_kernel_size=stage1_kernel_size,
            stage1_offset_scale=stage1_offset_scale,
        )
        self._freeze_existing_parameters()
        self._set_trainable_modes()
        self._printed_shapes = False

    def _freeze_existing_parameters(self) -> None:
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        for parameter in self.detector.p6_stage1_thin_line_branch.parameters():
            parameter.requires_grad = True

    def _set_trainable_modes(self) -> None:
        if not hasattr(self, "detector") or not hasattr(
            self.detector, "p6_stage1_thin_line_branch"
        ):
            return super()._set_trainable_modes()
        super()._set_trainable_modes()
        # 旧Stage 2分支冻结且无随机层；保持eval进一步明确冻结语义。
        self.detector.p5_thin_line_branch.eval()
        self.detector.p6_stage1_thin_line_branch.train()
        if not self.detector.training or not self.detector.transformer.training:
            raise RuntimeError("P6训练时SAM3根模块和Transformer必须处于train模式")
        if self.detector.backbone.training:
            raise RuntimeError("P6冻结Backbone必须处于eval模式")

    def configure_optimizers(self):
        gate = [self.detector.p6_stage1_thin_line_branch.gate]
        branch = [
            parameter
            for name, parameter in self.detector.p6_stage1_thin_line_branch.named_parameters()
            if name != "gate"
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": branch, "lr": self.hparams.stage1_branch_lr, "name": "stage1_dsconv"},
                {"params": gate, "lr": self.hparams.stage1_gate_lr, "name": "stage1_gate"},
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
        stage1 = self.detector.p6_stage1_thin_line_branch
        stage2 = self.detector.p5_thin_line_branch
        if not self._printed_shapes and stage1.last_input_shape is not None:
            print(
                "P6形状："
                f"Stage1={stage1.last_input_shape}，Stage1输出={stage1.last_output_shape}，"
                f"Stage2={stage2.last_input_shape}，Stage2输出={stage2.last_output_shape}，"
                f"Stage1 gate={float(stage1.gate.detach()):.8f}，"
                f"Stage2 gate={float(stage2.gate.detach()):.8f}",
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
        for name, parameter in self.detector.p6_stage1_thin_line_branch.named_parameters():
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
            self.log(f"train/grad_norm_stage1_{name}", value.sqrt(), on_step=True, on_epoch=False)

    def save_lora_checkpoint(self, path: Path) -> None:
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.detector.state_dict().items()
            if "parametrizations" in name
            or name.startswith(("dot_prod_scoring.", "segmentation_head."))
            or name.startswith(P5_BRANCH_PREFIX)
            or name.startswith(P6_BRANCH_PREFIX)
        }
        meta = {
            **self.p5_meta,
            "experiment": "TinyViT P6 frozen P5 plus Stage 1 and Stage 2 DSConv branches",
            "student_source": str(Path(self.hparams.p5_checkpoint)),
            "p6_multiscale_dsconv": True,
            "p6_stage1_branch_channels": int(self.hparams.stage1_branch_channels),
            "p6_stage1_kernel_size": int(self.hparams.stage1_kernel_size),
            "p6_stage1_offset_scale": float(self.hparams.stage1_offset_scale),
            "p6_stage1_branch_lr": float(self.hparams.stage1_branch_lr),
            "p6_stage1_gate_lr": float(self.hparams.stage1_gate_lr),
            "p6_frozen_p5": True,
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
    parser.add_argument("--stage1-branch-lr", type=float, default=1e-4)
    parser.add_argument("--stage1-gate-lr", type=float, default=1e-3)
    parser.add_argument("--stage1-branch-channels", type=int, default=64)
    parser.add_argument("--stage1-kernel-size", type=int, default=9)
    parser.add_argument("--stage1-offset-scale", type=float, default=1.0)
    parser.add_argument("--log-name", default="p6_stage1_frozen_p5")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(
        student_lora=EXP_DIR / "weights/p5a_dsconv_frozen_20ep.epoch15.pt",
        save=EXP_DIR / "weights/p6_stage1_frozen_p5.pt",
        epochs=10,
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
    model = P6MultiscaleDSConvModule(
        cache_root=args.cache_root,
        feature_cache_root=args.feature_cache_root,
        p5_checkpoint=args.student_lora,
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
        stage1_branch_lr=args.stage1_branch_lr,
        stage1_gate_lr=args.stage1_gate_lr,
        stage1_branch_channels=args.stage1_branch_channels,
        stage1_kernel_size=args.stage1_kernel_size,
        stage1_offset_scale=args.stage1_offset_scale,
    )
    trainable = [name for name, parameter in model.detector.named_parameters() if parameter.requires_grad]
    invalid = [name for name in trainable if not name.startswith(P6_BRANCH_PREFIX)]
    if invalid:
        raise RuntimeError(f"P6存在意外可训练旧参数：{invalid[:20]}")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    p5_count = sum(parameter.numel() for parameter in model.detector.p5_thin_line_branch.parameters())
    p6_count = sum(parameter.numel() for parameter in model.detector.p6_stage1_thin_line_branch.parameters())
    print(
        f"P6参数：总计={total:,}，冻结Stage2分支={p5_count:,}，"
        f"新增且可训练Stage1分支={p6_count:,}/{trainable_count:,}，"
        f"Stage1={model.attachment.stage1_channels}x{model.attachment.stage1_resolution}",
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
        limit_train_batches=2 if args.dry_run else 1.0,
        limit_val_batches=1 if args.dry_run else 1.0,
        log_every_n_steps=1 if args.dry_run else 10,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存P6最后权重：{args.save}", flush=True)


if __name__ == "__main__":
    main()
