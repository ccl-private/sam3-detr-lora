#!/usr/bin/env python3
"""训练P7：冻结P6，只训练直连高/中分辨率FPN的两个零门控适配器。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import lightning as L
import torch

P7_DIR = Path(__file__).resolve().parent
EXP_DIR = P7_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
P5_DIR = EXP_DIR / "p5_dsconv_thin_line"
P6_DIR = EXP_DIR / "p6_multiscale_dsconv"
for path in (PROJECT_ROOT, STAGE3_DIR, EXP_DIR, EXP_DIR / "p1_image_feature", P5_DIR, P6_DIR, P7_DIR):
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import train_p0_image_lora as p0
from dsconv_branch import P5_BRANCH_PREFIX
from highres_fpn import P7_BRANCH_PREFIX, restore_p6_then_attach_p7
from multiscale_dsconv import P6_BRANCH_PREFIX
from sam3_detr_exp.utils import CrackYoloSegDataModule
from train_p1_image_feature import P1ImageFeatureDistillModule


class P7HighResolutionFPNModule(P1ImageFeatureDistillModule):
    def __init__(
        self, *args, p6_checkpoint: Path,
        adapter_lr: float = 1e-4, gate_lr: float = 1e-3, **kwargs,
    ) -> None:
        super().__init__(*args, student_lora=p6_checkpoint, **kwargs)
        self.hparams.p6_checkpoint = str(p6_checkpoint)
        self.hparams.adapter_lr = adapter_lr
        self.hparams.gate_lr = gate_lr
        self.p6_meta, self.attachment = restore_p6_then_attach_p7(
            self.detector, p6_checkpoint,
        )
        self._freeze_existing_parameters()
        self._set_trainable_modes()
        self._printed_shapes = False

    def _freeze_existing_parameters(self) -> None:
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        for parameter in self.detector.p7_highres_fpn_adapters.parameters():
            parameter.requires_grad = True

    def _set_trainable_modes(self) -> None:
        if not hasattr(self, "detector") or not hasattr(self.detector, "p7_highres_fpn_adapters"):
            return super()._set_trainable_modes()
        super()._set_trainable_modes()
        self.detector.p5_thin_line_branch.eval()
        self.detector.p6_stage1_thin_line_branch.eval()
        self.detector.p7_highres_fpn_adapters.train()
        if not self.detector.training or not self.detector.transformer.training:
            raise RuntimeError("P7训练时SAM3根模块和Transformer必须处于train模式")
        if self.detector.backbone.training:
            raise RuntimeError("P7冻结Backbone必须处于eval模式")

    def configure_optimizers(self):
        gates, adapters = [], []
        for name, parameter in self.detector.p7_highres_fpn_adapters.named_parameters():
            (gates if name.endswith("gate") else adapters).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": adapters, "lr": self.hparams.adapter_lr, "name": "fpn_adapters"},
                {"params": gates, "lr": self.hparams.gate_lr, "name": "fpn_gates"},
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
        high = self.detector.p7_highres_fpn_adapters.high
        mid = self.detector.p7_highres_fpn_adapters.mid
        if not self._printed_shapes and high.last_input_shape is not None:
            print(
                f"P7形状：Stage1方向特征={high.last_input_shape} -> 高层={high.last_output_shape}，"
                f"Stage2方向特征={mid.last_input_shape} -> 中层={mid.last_output_shape}，"
                f"gate_high={float(high.gate.detach()):.8f}，gate_mid={float(mid.gate.detach()):.8f}",
                flush=True,
            )
            self._printed_shapes = True
        return result

    def on_after_backward(self) -> None:
        for level in ("high", "mid"):
            module = getattr(self.detector.p7_highres_fpn_adapters, level)
            for group, parameters in (
                ("projection", list(module.projection.parameters()) + list(module.norm.parameters())),
                ("gate", [module.gate]),
            ):
                value = torch.zeros((), device=self.device)
                for parameter in parameters:
                    if parameter.grad is not None:
                        value += parameter.grad.detach().float().norm(2).square()
                self.log(f"train/grad_norm_{level}_{group}", value.sqrt(), on_step=True, on_epoch=False)

    def save_lora_checkpoint(self, path: Path) -> None:
        prefixes = ("dot_prod_scoring.", "segmentation_head.", P5_BRANCH_PREFIX, P6_BRANCH_PREFIX, P7_BRANCH_PREFIX)
        state = {
            name: tensor.detach().cpu() for name, tensor in self.detector.state_dict().items()
            if "parametrizations" in name or name.startswith(prefixes)
        }
        meta = {
            **self.p6_meta,
            "experiment": "TinyViT P7 frozen P6 plus direct high/mid resolution FPN adapters",
            "student_source": str(Path(self.hparams.p6_checkpoint)),
            "p7_highres_fpn": True,
            "p7_high_fpn_index": 0,
            "p7_mid_fpn_index": 1,
            "p7_adapter_lr": float(self.hparams.adapter_lr),
            "p7_gate_lr": float(self.hparams.gate_lr),
            "p7_frozen_p6": True,
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
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--log-name", default="p7_highres_fpn_frozen_p6")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(
        student_lora=EXP_DIR / "weights/p6_stage1_frozen_p5.epoch8.pt",
        save=EXP_DIR / "weights/p7_highres_fpn_frozen_p6.pt",
        epochs=10,
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
    model = P7HighResolutionFPNModule(
        cache_root=args.cache_root, feature_cache_root=args.feature_cache_root,
        p6_checkpoint=args.student_lora, checkpoint=args.checkpoint,
        resolution=args.resolution, lora_lr=args.lora_lr, head_lr=args.head_lr,
        image_lora_lr=args.image_lora_lr, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        image_lora_rank=args.image_lora_rank, image_lora_alpha=args.image_lora_alpha,
        image_lora_dropout=args.image_lora_dropout,
        image_lora_stages=tuple(args.image_lora_stages), weight_decay=args.weight_decay,
        kd_weight=args.kd_weight, kd_warmup_ratio=args.kd_warmup_ratio,
        quality_threshold=args.quality_threshold, temperature=args.temperature,
        image_feature_kd_weight=args.image_feature_kd_weight,
        foreground_weight=args.foreground_weight,
        adapter_lr=args.adapter_lr, gate_lr=args.gate_lr,
    )
    trainable = [name for name, parameter in model.detector.named_parameters() if parameter.requires_grad]
    invalid = [name for name in trainable if not name.startswith(P7_BRANCH_PREFIX)]
    if invalid:
        raise RuntimeError(f"P7存在意外可训练旧参数：{invalid[:20]}")
    print(
        f"P7参数：总计={sum(p.numel() for p in model.parameters()):,}，"
        f"新增且可训练={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
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
        print(f"已保存P7最后权重：{args.save}", flush=True)


if __name__ == "__main__":
    main()
