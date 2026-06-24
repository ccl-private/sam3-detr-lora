#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import lightning as L
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sam3_detr_exp.model import DetrLoraLightningModule
from sam3_detr_exp.utils import CrackYoloSegDataModule
from sam3_detr_exp.utils.detr_lora_data import DEFAULT_DATASET_ROOT

EXP_ROOT = Path(__file__).resolve().parent
LORA_DIR = EXP_ROOT / "weights_lora"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train detector-only LoRA on the crack_segment YOLO segmentation dataset with Lightning 2.6.5."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
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
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--attn-only", action="store_true")
    parser.add_argument("--train-dot-score", action="store_true")
    parser.add_argument("--train-seg-head", action="store_true")
    parser.add_argument("--save", type=Path, default=LORA_DIR / "detr_transformer_lora.pt")
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    L.seed_everything(args.seed, workers=True)

    datamodule = CrackYoloSegDataModule(
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        val_split=args.val_split,
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
        f"dataset train_split={args.train_split} "
        f"samples={len(datamodule.train_dataset) if datamodule.train_dataset is not None else 0} "
        f"root={args.dataset_root}"
    )
    if datamodule.val_dataset is not None:
        print(
            f"dataset val_split={args.val_split} "
            f"samples={len(datamodule.val_dataset)} "
            f"root={args.dataset_root}"
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
    )
    print("trainable params:", sum(p.numel() for p in module.parameters() if p.requires_grad))
    print("attached lora modules:", len(module.attached_lora_modules))
    for name in module.attached_lora_modules[:12]:
        print("  ", name)
    if len(module.attached_lora_modules) > 12:
        print("  ...")

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=1 if args.dry_run else args.epochs,
        precision=args.precision,
        log_every_n_steps=args.log_every,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1 if args.dry_run else args.limit_train_batches,
        limit_val_batches=1 if args.dry_run else args.limit_val_batches,
        fast_dev_run=False,
    )
    trainer.fit(module, datamodule=datamodule)
    module.save_lora_checkpoint(args.save)
    print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
