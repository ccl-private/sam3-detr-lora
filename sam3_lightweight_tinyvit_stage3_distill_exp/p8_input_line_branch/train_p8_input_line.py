#!/usr/bin/env python3
"""训练P8：冻结P7，只训练输入侧504分辨率细线分支。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import lightning as L
import torch

P8_DIR = Path(__file__).resolve().parent
EXP_DIR = P8_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
EXTRA_DIRS = (
    PROJECT_ROOT, STAGE3_DIR, EXP_DIR, EXP_DIR / "p1_image_feature",
    EXP_DIR / "p5_dsconv_thin_line", EXP_DIR / "p6_multiscale_dsconv",
    EXP_DIR / "p7_highres_fpn", P8_DIR,
)
for path in EXTRA_DIRS:
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import train_p0_image_lora as p0
from dsconv_branch import P5_BRANCH_PREFIX
from highres_fpn import P7_BRANCH_PREFIX
from input_line_branch import P8_BRANCH_PREFIX, restore_p7_then_attach_p8
from multiscale_dsconv import P6_BRANCH_PREFIX
from sam3_detr_exp.utils import CrackYoloSegDataModule
from train_p1_image_feature import P1ImageFeatureDistillModule


class P8InputLineModule(P1ImageFeatureDistillModule):
    def __init__(
        self, *args, p7_checkpoint: Path,
        operator: str = "dsconv", stem_channels: int = 16,
        line_channels: int = 16, kernel_size: int = 9,
        offset_scale: float = 1.0, branch_lr: float = 1e-4,
        gate_lr: float = 1e-3, **kwargs,
    ) -> None:
        super().__init__(*args, student_lora=p7_checkpoint, **kwargs)
        self.hparams.p7_checkpoint = str(p7_checkpoint)
        self.hparams.operator = operator
        self.hparams.stem_channels = stem_channels
        self.hparams.line_channels = line_channels
        self.hparams.kernel_size = kernel_size
        self.hparams.offset_scale = offset_scale
        self.hparams.branch_lr = branch_lr
        self.hparams.gate_lr = gate_lr
        self.p7_meta = restore_p7_then_attach_p8(
            self.detector, p7_checkpoint, operator, stem_channels,
            line_channels, kernel_size, offset_scale,
        )
        self._freeze_existing_parameters()
        self._set_trainable_modes()
        self._printed_shapes = False

    def _freeze_existing_parameters(self) -> None:
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        for parameter in self.detector.p8_input_line_branch.parameters():
            parameter.requires_grad = True

    def _set_trainable_modes(self) -> None:
        if not hasattr(self, "detector") or not hasattr(self.detector, "p8_input_line_branch"):
            return super()._set_trainable_modes()
        super()._set_trainable_modes()
        self.detector.p5_thin_line_branch.eval()
        self.detector.p6_stage1_thin_line_branch.eval()
        self.detector.p7_highres_fpn_adapters.eval()
        self.detector.p8_input_line_branch.train()
        if not self.detector.training or not self.detector.transformer.training:
            raise RuntimeError("P8训练时SAM3根模块和Transformer必须处于train模式")
        if self.detector.backbone.training:
            raise RuntimeError("P8冻结Backbone必须处于eval模式")

    def configure_optimizers(self):
        gate = [self.detector.p8_input_line_branch.residual_gate]
        branch = [
            parameter for name, parameter in self.detector.p8_input_line_branch.named_parameters()
            if name != "residual_gate"
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": branch, "lr": self.hparams.branch_lr, "name": "input_line"},
                {"params": gate, "lr": self.hparams.gate_lr, "name": "residual_gate"},
            ], weight_decay=self.hparams.weight_decay,
        )

        def schedule(step: int) -> float:
            total = max(1, int(self.trainer.estimated_stepping_batches))
            warmup = max(1, int(total * self.hparams.kd_warmup_ratio))
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), "interval": "step",
        }}

    def _shared_step(self, batch, stage: str):
        result = super()._shared_step(batch, stage)
        branch = self.detector.p8_input_line_branch
        if not self._printed_shapes and branch.last_shapes:
            print(
                f"P8形状：{branch.last_shapes}，operator={branch.operator}，"
                f"gate={float(branch.residual_gate.detach()):.8f}", flush=True,
            )
            self._printed_shapes = True
        return result

    def on_after_backward(self) -> None:
        groups = {name: torch.zeros((), device=self.device) for name in (
            "stem", "direction", "refine", "semantic", "gate",
        )}
        for name, parameter in self.detector.p8_input_line_branch.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().norm(2).square()
            if name == "residual_gate": group = "gate"
            elif name.startswith("stem."): group = "stem"
            elif name.startswith(("horizontal.", "vertical.")): group = "direction"
            elif name.startswith("semantic_gate."): group = "semantic"
            else: group = "refine"
            groups[group] += value
        for name, value in groups.items():
            self.log(f"train/grad_norm_p8_{name}", value.sqrt(), on_step=True, on_epoch=False)

    def save_lora_checkpoint(self, path: Path) -> None:
        prefixes = (
            "dot_prod_scoring.", "segmentation_head.", P5_BRANCH_PREFIX,
            P6_BRANCH_PREFIX, P7_BRANCH_PREFIX, P8_BRANCH_PREFIX,
        )
        state = {
            name: tensor.detach().cpu() for name, tensor in self.detector.state_dict().items()
            if "parametrizations" in name or name.startswith(prefixes)
        }
        meta = {
            **self.p7_meta,
            "experiment": "TinyViT P8 frozen P7 plus 504-resolution input line branch",
            "student_source": str(Path(self.hparams.p7_checkpoint)),
            "p8_input_line_branch": True,
            "p8_operator": str(self.hparams.operator),
            "p8_stem_channels": int(self.hparams.stem_channels),
            "p8_line_channels": int(self.hparams.line_channels),
            "p8_kernel_size": int(self.hparams.kernel_size),
            "p8_offset_scale": float(self.hparams.offset_scale),
            "p8_branch_lr": float(self.hparams.branch_lr),
            "p8_gate_lr": float(self.hparams.gate_lr),
            "p8_frozen_p7": True,
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
    parser.add_argument("--operator", choices=("dsconv", "strip_conv"), default="dsconv")
    parser.add_argument("--stem-channels", type=int, default=16)
    parser.add_argument("--line-channels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--offset-scale", type=float, default=1.0)
    parser.add_argument("--branch-lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--log-name", default="p8_input_dsconv_frozen_p7")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(
        student_lora=EXP_DIR / "weights/p7_highres_fpn_frozen_p6.epoch9.pt",
        save=EXP_DIR / "weights/p8_input_dsconv_frozen_p7.pt", epochs=5,
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
        max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples,
    )
    datamodule.setup("fit")
    model = P8InputLineModule(
        cache_root=args.cache_root, feature_cache_root=args.feature_cache_root,
        p7_checkpoint=args.student_lora, checkpoint=args.checkpoint,
        resolution=args.resolution, lora_lr=args.lora_lr, head_lr=args.head_lr,
        image_lora_lr=args.image_lora_lr, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        image_lora_rank=args.image_lora_rank, image_lora_alpha=args.image_lora_alpha,
        image_lora_dropout=args.image_lora_dropout,
        image_lora_stages=tuple(args.image_lora_stages), weight_decay=args.weight_decay,
        kd_weight=args.kd_weight, kd_warmup_ratio=args.kd_warmup_ratio,
        quality_threshold=args.quality_threshold, temperature=args.temperature,
        image_feature_kd_weight=args.image_feature_kd_weight,
        foreground_weight=args.foreground_weight, operator=args.operator,
        stem_channels=args.stem_channels, line_channels=args.line_channels,
        kernel_size=args.kernel_size, offset_scale=args.offset_scale,
        branch_lr=args.branch_lr, gate_lr=args.gate_lr,
    )
    invalid = [
        name for name, parameter in model.detector.named_parameters()
        if parameter.requires_grad and not name.startswith(P8_BRANCH_PREFIX)
    ]
    if invalid:
        raise RuntimeError(f"P8存在意外可训练旧参数：{invalid[:20]}")
    print(
        f"P8参数：总计={sum(p.numel() for p in model.parameters()):,}，"
        f"新增且可训练={sum(p.numel() for p in model.parameters() if p.requires_grad):,}，"
        f"operator={args.operator}", flush=True,
    )
    best_path = args.best_save or p0.default_best_path(args.save)
    callbacks = [p0.SaveBest(best_path)]
    if args.save_every_epoch:
        callbacks.append(p0.SaveEveryEpoch(args.save))
    trainer = L.Trainer(
        default_root_dir=EXP_DIR / "logs" / args.log_name,
        accelerator=args.accelerator, devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs, precision=args.precision,
        gradient_clip_val=1.0, callbacks=callbacks, enable_checkpointing=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        limit_train_batches=2 if args.dry_run else 1.0,
        limit_val_batches=1 if args.dry_run else 1.0,
        log_every_n_steps=1 if args.dry_run else 10,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存P8最后权重：{args.save}", flush=True)
        if torch.cuda.is_available():
            print(
                f"P8训练峰值显存={torch.cuda.max_memory_allocated() / 2**20:.1f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
