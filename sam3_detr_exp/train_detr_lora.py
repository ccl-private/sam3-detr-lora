#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
import os
from pathlib import Path
import sys

import lightning as L
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sam3_detr_exp.model import DetrLoraLightningModule
from sam3_detr_exp.utils import CrackYoloSegDataModule
from sam3_detr_exp.utils.detr_lora_data import DEFAULT_DATA_YAML

EXP_ROOT = Path(__file__).resolve().parent
LORA_DIR = EXP_ROOT / "weights_lora"


def bind_local_cuda_device(accelerator: str) -> int | None:
    """Bind each re-launched Lightning DDP worker before SAM3 is constructed."""
    if accelerator != "gpu":
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_count = torch.cuda.device_count()
    if not 0 <= local_rank < device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the {device_count} visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    print(
        f"cuda binding: pid={os.getpid()} local_rank={local_rank} "
        f"current_device={torch.cuda.current_device()} "
        f"device={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )
    return local_rank


def default_best_path(last_path: Path) -> Path:
    return last_path.with_name(f"{last_path.stem}.best{last_path.suffix}")


class SaveBestLora(L.Callback):
    """Save a lightweight LoRA-only checkpoint whenever val/loss improves."""

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
                f"saved best: {self.output_path} "
                f"epoch={self.best_epoch} val/loss={self.best_val_loss:.6f}"
            )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train detector-only LoRA on a YAML-configured YOLO segmentation dataset with Lightning 2.6.5."
    )
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument(
        "--prompt-mode", type=str, default="class_name", choices=["class_name", "generic"]
    )
    parser.add_argument("--generic-prompt", type=str, default="crack")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-weight", type=float, default=2.0)
    parser.add_argument(
        "--loss-mode", choices=["simple", "sam3"], default="simple",
        help="Use the legacy simplified loss or SAM3 native IABCE/presence/focal-mask/Dice losses.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--attn-only", action="store_true")
    parser.add_argument("--train-dot-score", action="store_true")
    parser.add_argument("--train-seg-head", action="store_true")
    parser.add_argument("--save", type=Path, default=LORA_DIR / "detr_transformer_lora.pt")
    parser.add_argument(
        "--best-save", type=Path, default=None,
        help="Best-val LoRA path (default: insert .best before --save suffix).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32-true"],
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if args.accelerator == "gpu":
        bind_local_cuda_device(args.accelerator)
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
    print(
        f"train samples={len(datamodule.train_dataset) if datamodule.train_dataset is not None else 0} "
        f"mode={'multi_prompt' if datamodule.train_dataset and datamodule.train_dataset.multi_prompt else 'single_prompt'} "
        f"yaml={args.data_yaml}"
    )
    if datamodule.val_dataset is not None:
        print(
            f"val samples={len(datamodule.val_dataset)} "
            f"yaml={args.data_yaml}"
        )

    module = DetrLoraLightningModule(
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
    print("trainable params:", sum(p.numel() for p in module.parameters() if p.requires_grad))
    print("attached lora modules:", len(module.attached_lora_modules))
    for name in module.attached_lora_modules[:12]:
        print("  ", name)
    if len(module.attached_lora_modules) > 12:
        print("  ...")

    best_save = args.best_save or default_best_path(args.save)
    best_callback = SaveBestLora(best_save)

    trainer = L.Trainer(
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
        fast_dev_run=False,
    )
    trainer.fit(module, datamodule=datamodule)
    if trainer.is_global_zero:
        module.save_lora_checkpoint(args.save)
        print(f"saved last: {args.save}")
        if best_callback.best_epoch >= 0:
            print(
                f"best: {best_save} epoch={best_callback.best_epoch} "
                f"val/loss={best_callback.best_val_loss:.6f}"
            )


if __name__ == "__main__":
    main()
