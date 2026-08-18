#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path
import sys

import lightning as L
import torch

EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam3_lightweight_exp.bootstrap import activate_efficientsam3

activate_efficientsam3()

import sam3_detr_exp.model.detr_lora_module as module_impl
from sam3_detr_exp.utils import CrackYoloSegDataModule
from sam3_lightweight_exp.model_adapter import (
    DEFAULT_PRETRAINED,
    build_trainable_lightweight_detector,
)


def bind_local_cuda_device(accelerator: str) -> None:
    if accelerator != "gpu":
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    print(
        f"cuda binding: pid={os.getpid()} local_rank={local_rank} "
        f"device={torch.cuda.current_device()}",
        flush=True,
    )


def default_best_path(last_path: Path) -> Path:
    return last_path.with_name(f"{last_path.stem}.best{last_path.suffix}")


class SaveBestLora(L.Callback):
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.best_val_loss = float("inf")
        self.best_epoch = -1

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
        self.best_epoch = trainer.current_epoch
        if trainer.is_global_zero:
            pl_module.save_lora_checkpoint(self.output_path)
            print(
                f"saved best: {self.output_path} epoch={self.best_epoch} "
                f"val/loss={self.best_val_loss:.6f}",
                flush=True,
            )


def parser() -> ArgumentParser:
    result = ArgumentParser(description="LoRA fine-tuning for fixed-vocabulary TinyViT EfficientSAM3")
    result.add_argument("--data-yaml", type=Path, default=EXP_ROOT / "configs/roadline_lora.yaml")
    result.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    result.add_argument("--prompt-mode", default="class_name", choices=["class_name", "generic"])
    result.add_argument("--generic-prompt", default="crack")
    result.add_argument("--max-train-samples", type=int)
    result.add_argument("--max-val-samples", type=int)
    result.add_argument("--resolution", type=int, default=1008)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--lr", type=float, default=2e-4)
    result.add_argument("--weight-decay", type=float, default=1e-2)
    result.add_argument("--mask-weight", type=float, default=2.0)
    result.add_argument("--loss-mode", choices=["simple", "sam3"], default="sam3")
    result.add_argument("--lora-rank", type=int, default=8)
    result.add_argument("--lora-alpha", type=float, default=16.0)
    result.add_argument("--lora-dropout", type=float, default=0.05)
    result.add_argument("--decoder-only", action="store_true")
    result.add_argument("--attn-only", action="store_true")
    result.add_argument("--train-dot-score", action="store_true")
    result.add_argument("--train-seg-head", action="store_true")
    result.add_argument("--save", type=Path, default=EXP_ROOT / "weights_lora/roadline_tinyvit_s.pt")
    result.add_argument("--best-save", type=Path)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--devices", type=int, default=1)
    result.add_argument("--accelerator", default="gpu")
    result.add_argument("--precision", default="bf16-mixed", choices=["bf16-mixed", "16-mixed", "32-true"])
    result.add_argument("--log-every", type=int, default=10)
    result.add_argument("--limit-train-batches", type=float, default=1.0)
    result.add_argument("--limit-val-batches", type=float, default=1.0)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
    print(f"train samples={len(datamodule.train_dataset)} val samples={len(datamodule.val_dataset)}")

    module_impl.assert_modular_weights_exist = lambda: None
    module_impl.build_trainable_detector = partial(
        build_trainable_lightweight_detector,
        pretrained_path=args.pretrained,
    )
    model = module_impl.DetrLoraLightningModule(
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
    print(f"total params={sum(p.numel() for p in model.parameters()):,}")
    print(f"trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_path = args.best_save or default_best_path(args.save)
    best_callback = SaveBestLora(best_path)
    trainer = L.Trainer(
        default_root_dir=EXP_ROOT / "lightning_logs",
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs,
        precision=args.precision,
        log_every_n_steps=args.log_every,
        callbacks=[best_callback],
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1 if args.dry_run else args.limit_train_batches,
        limit_val_batches=1 if args.dry_run else args.limit_val_batches,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"saved last: {args.save}")


if __name__ == "__main__":
    main()
