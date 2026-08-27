#!/usr/bin/env python3
"""P9：官方TinyViT起点、P8完整结构、新Base教师，从头蒸馏20轮。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import lightning as L
import torch

P9_DIR = Path(__file__).resolve().parent
EXP_DIR = P9_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
EXTRA_DIRS = (
    PROJECT_ROOT, STAGE3_DIR, EXP_DIR, EXP_DIR / "p1_image_feature",
    EXP_DIR / "p5_dsconv_thin_line", EXP_DIR / "p6_multiscale_dsconv",
    EXP_DIR / "p7_highres_fpn", EXP_DIR / "p8_input_line_branch", P9_DIR,
)
for path in EXTRA_DIRS:
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import train_p0_image_lora as p0
from complete_p8_structure import attach_complete_p8_structure
from dsconv_branch import P5_BRANCH_PREFIX
from highres_fpn import P7_BRANCH_PREFIX
from image_lora import IMAGE_LORA_PREFIX
from input_line_branch import P8_BRANCH_PREFIX
from multiscale_dsconv import P6_BRANCH_PREFIX
from sam3_detr_exp.utils import CrackYoloSegDataModule
from train_p1_image_feature import P1ImageFeatureDistillModule


BRANCH_PREFIXES = (P5_BRANCH_PREFIX, P6_BRANCH_PREFIX, P7_BRANCH_PREFIX, P8_BRANCH_PREFIX)


class P9FreshP8Module(P1ImageFeatureDistillModule):
    def __init__(
        self, *args, branch_lr: float = 1e-4, gate_lr: float = 1e-3,
        p5_branch_channels: int = 128, p6_branch_channels: int = 64,
        kernel_size: int = 9, offset_scale: float = 1.0,
        p8_operator: str = "dsconv", p8_stem_channels: int = 16,
        p8_line_channels: int = 16, **kwargs,
    ) -> None:
        super().__init__(*args, student_lora=None, **kwargs)
        self.hparams.branch_lr = branch_lr
        self.hparams.gate_lr = gate_lr
        self.hparams.p5_branch_channels = p5_branch_channels
        self.hparams.p6_branch_channels = p6_branch_channels
        self.hparams.kernel_size = kernel_size
        self.hparams.offset_scale = offset_scale
        self.hparams.p8_operator = p8_operator
        self.hparams.p8_stem_channels = p8_stem_channels
        self.hparams.p8_line_channels = p8_line_channels
        attach_complete_p8_structure(
            self.detector, p5_branch_channels=p5_branch_channels,
            p6_branch_channels=p6_branch_channels, kernel_size=kernel_size,
            offset_scale=offset_scale, p8_operator=p8_operator,
            p8_stem_channels=p8_stem_channels,
            p8_line_channels=p8_line_channels,
        )
        self._set_trainable_modes()
        self._gradient_reports = 0

    def _set_trainable_modes(self) -> None:
        if not hasattr(self, "detector"):
            return
        super()._set_trainable_modes()
        for name in (
            "p5_thin_line_branch", "p6_stage1_thin_line_branch",
            "p7_highres_fpn_adapters", "p8_input_line_branch",
        ):
            module = getattr(self.detector, name, None)
            if module is not None:
                module.train()

    @staticmethod
    def _is_gate(name: str) -> bool:
        return name.endswith((".gate", ".residual_gate"))

    def configure_optimizers(self):
        groups = {key: [] for key in ("detr_lora", "image_lora", "heads", "branches", "gates")}
        for name, parameter in self.detector.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(("dot_prod_scoring.", "segmentation_head.")):
                groups["heads"].append(parameter)
            elif name.startswith(IMAGE_LORA_PREFIX):
                groups["image_lora"].append(parameter)
            elif name.startswith(BRANCH_PREFIXES):
                groups["gates" if self._is_gate(name) else "branches"].append(parameter)
            elif "parametrizations" in name:
                groups["detr_lora"].append(parameter)
            else:
                raise RuntimeError(f"P9发现未分类可训练参数：{name}")
        rates = {
            "detr_lora": self.hparams.lora_lr,
            "image_lora": self.hparams.image_lora_lr,
            "heads": self.hparams.head_lr,
            "branches": self.hparams.branch_lr,
            "gates": self.hparams.gate_lr,
        }
        optimizer = torch.optim.AdamW(
            [{"params": values, "lr": rates[name], "name": name}
             for name, values in groups.items() if values],
            weight_decay=self.hparams.weight_decay,
        )

        def schedule(step: int) -> float:
            total = max(1, int(self.trainer.estimated_stepping_batches))
            warmup = max(1, int(total * self.hparams.kd_warmup_ratio))
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
            "interval": "step",
        }}

    def on_after_backward(self) -> None:
        groups = {
            key: torch.zeros((), device=self.device)
            for key in ("detr_lora", "image_lora", "heads", "p5", "p6", "p7", "p8")
        }
        for name, parameter in self.detector.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().norm(2).square()
            if name.startswith(P5_BRANCH_PREFIX): group = "p5"
            elif name.startswith(P6_BRANCH_PREFIX): group = "p6"
            elif name.startswith(P7_BRANCH_PREFIX): group = "p7"
            elif name.startswith(P8_BRANCH_PREFIX): group = "p8"
            elif name.startswith(("dot_prod_scoring.", "segmentation_head.")): group = "heads"
            elif name.startswith(IMAGE_LORA_PREFIX): group = "image_lora"
            else: group = "detr_lora"
            groups[group] += value
        norms = {name: value.sqrt() for name, value in groups.items()}
        for name, value in norms.items():
            self.log(f"train/grad_norm_{name}", value, on_step=True, on_epoch=False)
        if self._gradient_reports < 2:
            print(
                "P9梯度检查 step=" + str(self.global_step) + " "
                + " ".join(f"{name}={float(value):.6g}" for name, value in norms.items()),
                flush=True,
            )
            self._gradient_reports += 1

    def on_fit_end(self) -> None:
        if self.device.type == "cuda":
            print(
                f"P9单卡峰值显存：allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB，"
                f"reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
                flush=True,
            )

    def save_lora_checkpoint(self, path: Path) -> None:
        prefixes = ("dot_prod_scoring.", "segmentation_head.", *BRANCH_PREFIXES)
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.detector.state_dict().items()
            if "parametrizations" in name or name.startswith(prefixes)
        }
        meta = {
            **self.student_source_meta,
            "experiment": "P9 fresh TinyViT plus complete P8 structure and new teacher",
            "teacher_cache": str(Path(self.hparams.cache_root)),
            "feature_cache_root": str(Path(self.hparams.feature_cache_root)),
            "num_generic_negatives": 0,
            "lora_rank": int(self.hparams.lora_rank),
            "lora_alpha": float(self.hparams.lora_alpha),
            "lora_dropout": float(self.hparams.lora_dropout),
            "decoder_only": False,
            "attn_only": False,
            "train_dot_score": True,
            "train_seg_head": True,
            "image_lora_rank": int(self.hparams.image_lora_rank),
            "image_lora_alpha": float(self.hparams.image_lora_alpha),
            "image_lora_dropout": float(self.hparams.image_lora_dropout),
            "image_lora_stages": list(self.hparams.image_lora_stages),
            "p5_stage": 2,
            "p5_branch_channels": int(self.hparams.p5_branch_channels),
            "p5_dsconv_kernel_size": int(self.hparams.kernel_size),
            "p5_offset_scale": float(self.hparams.offset_scale),
            "p6_multiscale_dsconv": True,
            "p6_stage1_branch_channels": int(self.hparams.p6_branch_channels),
            "p6_stage1_kernel_size": int(self.hparams.kernel_size),
            "p6_stage1_offset_scale": float(self.hparams.offset_scale),
            "p7_highres_fpn": True,
            "p8_input_line_branch": True,
            "p8_operator": str(self.hparams.p8_operator),
            "p8_stem_channels": int(self.hparams.p8_stem_channels),
            "p8_line_channels": int(self.hparams.p8_line_channels),
            "p8_kernel_size": int(self.hparams.kernel_size),
            "p8_offset_scale": float(self.hparams.offset_scale),
            "p9_fresh_student": True,
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
    parser.add_argument("--p5-branch-channels", type=int, default=128)
    parser.add_argument("--p6-branch-channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--offset-scale", type=float, default=1.0)
    parser.add_argument("--p8-operator", choices=("dsconv", "strip_conv"), default="dsconv")
    parser.add_argument("--p8-stem-channels", type=int, default=16)
    parser.add_argument("--p8-line-channels", type=int, default=16)
    parser.add_argument("--log-name", default="p9_fresh_p8_new_teacher")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(
        data_yaml=P9_DIR / "configs/roadline_no_generic_negatives.yaml",
        cache_root=P9_DIR / "cache/new_teacher_outputs",
        save=EXP_DIR / "weights/p9_fresh_p8_new_teacher.pt",
        epochs=20, batch_size=4,
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
        num_generic_negatives=0,
    )
    datamodule.setup("fit")
    if datamodule.train_dataset is None or datamodule.train_dataset.num_generic_negatives != 0:
        raise RuntimeError("P9必须关闭全部域外通用负提示")
    model = P9FreshP8Module(
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
        p5_branch_channels=args.p5_branch_channels,
        p6_branch_channels=args.p6_branch_channels,
        kernel_size=args.kernel_size, offset_scale=args.offset_scale,
        p8_operator=args.p8_operator,
        p8_stem_channels=args.p8_stem_channels,
        p8_line_channels=args.p8_line_channels,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"P9起点=官方TinyViT Stage-3，域外负提示=0，"
        f"总参数={sum(p.numel() for p in model.parameters()):,}，"
        f"可训练参数={trainable:,}", flush=True,
    )
    callbacks = [p0.SaveBest(args.best_save or p0.default_best_path(args.save))]
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
        print(f"已保存P9最后权重：{args.save}", flush=True)


if __name__ == "__main__":
    main()
