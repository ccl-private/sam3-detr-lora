#!/usr/bin/env python3
"""使用 Stage-3 EV-M 与实时 MobileCLIP-S0 文本特征进行道路标线 LoRA 微调。"""

from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path
import sys

import lightning as L
import torch

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import activate_efficientsam3

activate_efficientsam3()

import sam3_detr_exp.model.detr_lora_module as module_impl
from model_adapter import DEFAULT_STAGE3_CHECKPOINT, build_trainable_stage3_detector
from sam3_detr_exp.utils import CrackYoloSegDataModule
from sam3_detr_exp.utils.detr_lora_utils import save_lora_state


def bind_local_cuda_device(accelerator: str) -> None:
    if accelerator != "gpu":
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    print(
        f"CUDA 绑定：进程={os.getpid()} 本地序号={local_rank} "
        f"设备={torch.cuda.current_device()}",
        flush=True,
    )


def default_best_path(last_path: Path) -> Path:
    return last_path.with_name(f"{last_path.stem}.best{last_path.suffix}")


class Stage3LoraModule(module_impl.DetrLoraLightningModule):
    def save_lora_checkpoint(self, output_path: Path) -> None:
        meta = {
            "base_model": "EfficientSAM3 Stage-3 EV-M",
            "base_checkpoint": str(self.detector.stage3_lora_metadata["base_checkpoint"]),
            "text_features": self.detector.stage3_lora_metadata["text_features"],
            "lora_rank": self.hparams.lora_rank,
            "lora_alpha": self.hparams.lora_alpha,
            "lora_dropout": self.hparams.lora_dropout,
            "decoder_only": self.hparams.decoder_only,
            "attn_only": self.hparams.attn_only,
            "train_dot_score": self.hparams.train_dot_score,
            "train_seg_head": self.hparams.train_seg_head,
        }
        save_lora_state(self.detector, output_path, meta)


class SaveBestLora(L.Callback):
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val/loss")
        if value is None:
            return
        val_loss = float(value.detach().cpu())
        if val_loss >= self.best_val_loss:
            return
        self.best_val_loss = val_loss
        if trainer.is_global_zero:
            pl_module.save_lora_checkpoint(self.output_path)
            print(
                f"已保存最佳权重：{self.output_path} "
                f"轮次={trainer.current_epoch} 验证损失={val_loss:.6f}",
                flush=True,
            )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=EXP_DIR / "configs/roadline_lora.yaml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE3_CHECKPOINT)
    parser.add_argument("--text-mode", choices=["runtime", "precomputed"], default="runtime")
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--prompt-mode", default="class_name", choices=["class_name", "generic"])
    parser.add_argument("--generic-prompt", default="road marking")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-weight", type=float, default=2.0)
    parser.add_argument("--loss-mode", choices=["simple", "sam3"], default="sam3")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--attn-only", action="store_true")
    parser.add_argument("--train-dot-score", action="store_true")
    parser.add_argument("--train-seg-head", action="store_true")
    parser.add_argument("--save", type=Path, default=EXP_DIR / "weights_lora/roadline_stage3_ev_m.pt")
    parser.add_argument("--best-save", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default="bf16-mixed", choices=["bf16-mixed", "16-mixed", "32-true"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.text_mode == "precomputed" and args.text_cache is None:
        raise ValueError("--text-mode precomputed 必须同时提供 --text-cache")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 GPU，但 CUDA 不可用")
    bind_local_cuda_device(args.accelerator)
    if args.accelerator == "gpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    L.seed_everything(args.seed, workers=True)

    datamodule = CrackYoloSegDataModule(
        data_yaml=args.data_yaml,
        resolution=args.resolution,
        prompt_mode=args.prompt_mode,
        generic_prompt=args.generic_prompt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    datamodule.setup("fit")
    print(f"训练样本={len(datamodule.train_dataset)} 验证样本={len(datamodule.val_dataset)}")
    print(f"文本特征模式={args.text_mode}")

    module_impl.assert_modular_weights_exist = lambda: None
    module_impl.build_trainable_detector = partial(
        build_trainable_stage3_detector,
        checkpoint_path=args.checkpoint,
        text_mode=args.text_mode,
        text_cache_path=args.text_cache,
    )
    model = Stage3LoraModule(
        resolution=args.resolution,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_weight=args.mask_weight,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        decoder_only=args.decoder_only,
        attn_only=args.attn_only,
        train_dot_score=args.train_dot_score,
        train_seg_head=args.train_seg_head,
        loss_mode=args.loss_mode,
    )
    print(f"总参数量={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"可训练参数量={sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}")

    best_path = args.best_save or default_best_path(args.save)
    trainer = L.Trainer(
        default_root_dir=EXP_DIR / "lightning_logs",
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs,
        precision=args.precision,
        log_every_n_steps=args.log_every,
        callbacks=[SaveBestLora(best_path)],
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1 if args.dry_run else args.limit_train_batches,
        limit_val_batches=1 if args.dry_run else args.limit_val_batches,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存最后权重：{args.save}")


if __name__ == "__main__":
    main()
