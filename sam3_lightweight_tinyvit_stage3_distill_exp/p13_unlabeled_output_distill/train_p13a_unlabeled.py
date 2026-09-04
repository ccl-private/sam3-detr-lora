#!/usr/bin/env python3
"""P13-A：在P12有标签训练之外加入单教师无标签高置信输出蒸馏。"""
from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

P13_DIR = Path(__file__).resolve().parent
EXP_DIR = P13_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
for path in (
    PROJECT_ROOT, STAGE3_DIR, EXP_DIR, EXP_DIR / "p1_image_feature",
    EXP_DIR / "p5_dsconv_thin_line", EXP_DIR / "p6_multiscale_dsconv",
    EXP_DIR / "p7_highres_fpn", EXP_DIR / "p8_input_line_branch",
    EXP_DIR / "p9_fresh_p8_new_teacher", EXP_DIR / "p12_query_set_distill", P13_DIR,
):
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3
activate_efficientsam3()

import train_p0_image_lora as p0
from common import cache_file, load_cache, prompt_key
from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from sam3_detr_exp.model.detr_lora_module import _external_matching, _move_tensors_to_device, _reset_stale_decoder_caches
from sam3_detr_exp.utils import CrackYoloSegDataModule, build_prompt, make_find_stage
from train_p12_query_set import P12QuerySetModule, verify_dense_cache
from unlabeled_data import UnlabeledRoadlineDataset, collate_unlabeled


class P13AUnlabeledModule(P12QuerySetModule):
    def __init__(
        self, *args, unlabeled_cache_root: Path, prompts: list[str],
        unlabeled_kd_weight: float = 0.5, unlabeled_kd_warmup_ratio: float = 0.10,
        unlabeled_presence_weight: float = 1.0, unlabeled_class_weight: float = 1.0,
        unlabeled_box_l1_weight: float = 5.0, unlabeled_giou_weight: float = 2.0,
        unlabeled_mask_bce_weight: float = 2.0, unlabeled_mask_dice_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hparams.unlabeled_cache_root = str(unlabeled_cache_root)
        self.hparams.unlabeled_prompts = list(prompts)
        for name, value in {
            "unlabeled_kd_weight": unlabeled_kd_weight,
            "unlabeled_kd_warmup_ratio": unlabeled_kd_warmup_ratio,
            "unlabeled_presence_weight": unlabeled_presence_weight,
            "unlabeled_class_weight": unlabeled_class_weight,
            "unlabeled_box_l1_weight": unlabeled_box_l1_weight,
            "unlabeled_giou_weight": unlabeled_giou_weight,
            "unlabeled_mask_bce_weight": unlabeled_mask_bce_weight,
            "unlabeled_mask_dice_weight": unlabeled_mask_dice_weight,
        }.items():
            self.hparams[name] = value

    def unlabeled_scale(self) -> float:
        if self.trainer is None:
            return float(self.hparams.unlabeled_kd_weight)
        warmup = max(1, int(self.trainer.estimated_stepping_batches * float(self.hparams.unlabeled_kd_warmup_ratio)))
        ratio = min(1.0, (self.global_step + 1) / warmup)
        return float(self.hparams.unlabeled_kd_weight) * ratio

    @staticmethod
    def _soft_dice(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student, teacher = student_logits.sigmoid(), teacher_logits.sigmoid()
        intersection = (student * teacher).flatten(1).sum(1)
        denominator = student.flatten(1).sum(1) + teacher.flatten(1).sum(1)
        return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

    def _unlabeled_loss(self, outputs: dict, caches: list[dict]) -> tuple[torch.Tensor, dict]:
        device = outputs["pred_logits"].device
        zero = outputs["pred_logits"].sum() * 0.0
        presence_losses, class_losses, l1_losses, giou_losses = [], [], [], []
        mask_bce_losses, mask_dice_losses, instance_weights = [], [], []
        presence = outputs.get("presence_logit_dec")
        prompt_count = len(self.hparams.unlabeled_prompts)
        matched = 0
        for image_index, payload in enumerate(caches):
            for prompt_index, text in enumerate(self.hparams.unlabeled_prompts):
                output_index = image_index * prompt_count + prompt_index
                entry = payload["prompts"][prompt_key(text)]
                teacher_presence = entry["presence_logit"].to(device=device, dtype=torch.float32).reshape(())
                if presence is not None:
                    student_presence = presence[output_index].float().reshape(())
                    temp = float(self.hparams.temperature)
                    presence_losses.append(F.binary_cross_entropy_with_logits(
                        student_presence / temp, (teacher_presence / temp).sigmoid()) * temp**2)
                instances = entry["instances"]
                if not instances:
                    continue
                teacher_boxes = torch.stack([item["box"] for item in instances]).to(device=device, dtype=torch.float32)
                student_boxes = outputs["pred_boxes"][output_index].float()
                teacher_logits = torch.stack([item["query_logit"].reshape(()) for item in instances]).to(device=device, dtype=torch.float32)
                student_logits = outputs["pred_logits"][output_index].float().reshape(-1)
                giou_matrix = generalized_box_iou(box_cxcywh_to_xyxy(teacher_boxes), box_cxcywh_to_xyxy(student_boxes))
                class_cost = F.binary_cross_entropy_with_logits(
                    student_logits.unsqueeze(0).expand(len(instances), -1),
                    teacher_logits.sigmoid().unsqueeze(1).expand(-1, len(student_logits)),
                    reduction="none",
                )
                teacher_masks = torch.stack([item["mask_logit"] for item in instances]).to(device=device, dtype=torch.float32).sigmoid()
                student_masks = outputs["pred_masks"][output_index].float().sigmoid()
                teacher_small = F.interpolate(teacher_masks.unsqueeze(1), size=(36, 36), mode="bilinear", align_corners=False).flatten(1)
                student_small = F.interpolate(student_masks.unsqueeze(1), size=(36, 36), mode="bilinear", align_corners=False).flatten(1)
                intersection = teacher_small @ student_small.t()
                mask_cost = 1.0 - (2.0 * intersection + 1.0) / (
                    teacher_small.sum(1, keepdim=True) + student_small.sum(1).unsqueeze(0) + 1.0
                )
                cost = (
                    class_cost + 5.0 * torch.cdist(teacher_boxes, student_boxes, p=1)
                    + 2.0 * (1.0 - giou_matrix) + 2.0 * mask_cost
                ).detach().cpu().numpy()
                teacher_ids, student_ids = linear_sum_assignment(cost)
                for teacher_id, student_id in zip(teacher_ids.tolist(), student_ids.tolist()):
                    item = instances[teacher_id]
                    score = item["combined_score"].to(device=device, dtype=torch.float32).sqrt().clamp(min=1e-3)
                    teacher_logit = item["query_logit"].to(device=device, dtype=torch.float32).reshape(())
                    student_logit = outputs["pred_logits"][output_index, student_id].float().reshape(())
                    temp = float(self.hparams.temperature)
                    class_losses.append(F.binary_cross_entropy_with_logits(
                        student_logit / temp, (teacher_logit / temp).sigmoid()) * temp**2)
                    teacher_box = teacher_boxes[teacher_id]
                    student_box = student_boxes[student_id]
                    l1_losses.append(F.l1_loss(student_box, teacher_box))
                    giou_losses.append(1.0 - generalized_box_iou(
                        box_cxcywh_to_xyxy(student_box.unsqueeze(0)),
                        box_cxcywh_to_xyxy(teacher_box.unsqueeze(0)),
                    )[0, 0])
                    teacher_mask = item["mask_logit"].to(device=device, dtype=torch.float32).unsqueeze(0)
                    student_mask = outputs["pred_masks"][output_index, student_id].float().unsqueeze(0)
                    mask_bce_losses.append(F.binary_cross_entropy_with_logits(student_mask, teacher_mask.sigmoid()))
                    mask_dice_losses.append(self._soft_dice(student_mask, teacher_mask)[0])
                    instance_weights.append(score)
                    matched += 1

        presence_loss = torch.stack(presence_losses).mean() if presence_losses else zero
        if instance_weights:
            weights = torch.stack(instance_weights)
            weights = weights / weights.sum().clamp(min=1e-6)
            weighted = lambda values: (weights * torch.stack(values)).sum()
            class_loss, l1_loss, giou_loss = weighted(class_losses), weighted(l1_losses), weighted(giou_losses)
            mask_bce, mask_dice = weighted(mask_bce_losses), weighted(mask_dice_losses)
        else:
            class_loss = l1_loss = giou_loss = mask_bce = mask_dice = zero
        total = (
            float(self.hparams.unlabeled_presence_weight) * presence_loss
            + float(self.hparams.unlabeled_class_weight) * class_loss
            + float(self.hparams.unlabeled_box_l1_weight) * l1_loss
            + float(self.hparams.unlabeled_giou_weight) * giou_loss
            + float(self.hparams.unlabeled_mask_bce_weight) * mask_bce
            + float(self.hparams.unlabeled_mask_dice_weight) * mask_dice
        )
        return total, {
            "presence": presence_loss.detach(), "class": class_loss.detach(),
            "box_l1": l1_loss.detach(), "giou": giou_loss.detach(),
            "mask_bce": mask_bce.detach(), "mask_dice": mask_dice.detach(),
            "matches": torch.tensor(float(matched), device=device),
        }

    def _unlabeled_step(self, batch: dict) -> torch.Tensor:
        images = batch["images"].to(self.device, non_blocking=True)
        image_paths = batch["image_paths"]
        prompts = list(self.hparams.unlabeled_prompts)
        texts = prompts * len(image_paths)
        image_ids = [image_index for image_index in range(len(image_paths)) for _ in prompts]
        caches = [load_cache(Path(self.hparams.unlabeled_cache_root), "train", path) for path in image_paths]
        _reset_stale_decoder_caches(self.detector, self.device)
        backbone_out = self.detector.backbone.forward_image(images)
        with torch.no_grad():
            backbone_out.update(self.detector.backbone.forward_text(texts, device=self.device))
        backbone_out = _move_tensors_to_device(backbone_out, self.device)
        with _external_matching(self.detector):
            outputs = self.detector.forward_grounding(
                backbone_out=backbone_out,
                find_input=make_find_stage(torch.tensor(image_ids), self.device),
                find_target=None,
                geometric_prompt=build_prompt(self.detector, len(texts), self.device),
            )
        loss, metrics = self._unlabeled_loss(outputs, caches)
        for name, value in metrics.items():
            self.log(f"train/unlabeled_{name}", value, batch_size=len(image_paths), sync_dist=False)
        self.log("train/unlabeled_kd", loss.detach(), batch_size=len(image_paths), prog_bar=True, sync_dist=False)
        return loss

    def training_step(self, batch, batch_idx):
        labeled_loss = self._shared_step(batch["labeled"], "train")
        unlabeled_loss = self._unlabeled_step(batch["unlabeled"])
        scale = self.unlabeled_scale()
        total = labeled_loss + scale * unlabeled_loss
        self.log("train/unlabeled_scale", scale, batch_size=len(batch["unlabeled"]["image_paths"]), sync_dist=False)
        self.log("train/combined_loss", total.detach(), prog_bar=True, batch_size=len(batch["labeled"]), sync_dist=False)
        return total

    def save_lora_checkpoint(self, path):
        super().save_lora_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["meta"].update({
            "experiment": "P13-A P12 plus single-teacher unlabeled high-confidence output KD",
            "unlabeled_cache_root": self.hparams.unlabeled_cache_root,
            "unlabeled_kd_weight": float(self.hparams.unlabeled_kd_weight),
            "unlabeled_kd_warmup_ratio": float(self.hparams.unlabeled_kd_warmup_ratio),
        })
        torch.save(payload, path)


def verify_unlabeled_cache(dataset: UnlabeledRoadlineDataset, root: Path, prompts: list[str]) -> None:
    expected = {prompt_key(text) for text in prompts}
    missing, invalid = [], []
    for row in dataset.records:
        path = cache_file(root, "train", row["image_path"])
        if not path.exists():
            missing.append(path)
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if set(payload.get("prompts", {})) != expected:
            invalid.append(path)
    if missing or invalid:
        raise RuntimeError(f"P13无标签缓存未完成：缺少{len(missing)}，格式错误{len(invalid)}；请先执行缓存脚本")


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
    parser.add_argument("--candidate-set-kd-weight", type=float, default=0.20)
    parser.add_argument("--candidate-topk", type=int, default=50)
    parser.add_argument("--candidate-min-score", type=float, default=0.05)
    parser.add_argument("--candidate-rank-weight", type=float, default=0.10)
    parser.add_argument("--relation-kd-weight", type=float, default=0.10)
    parser.add_argument("--relation-iou-scale", type=float, default=4.0)
    parser.add_argument("--unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--unlabeled-cache-root", type=Path, required=True)
    parser.add_argument("--unlabeled-batch-size", type=int, default=4)
    parser.add_argument("--max-unlabeled-samples", type=int)
    parser.add_argument("--unlabeled-kd-weight", type=float, default=0.5)
    parser.add_argument("--unlabeled-kd-warmup-ratio", type=float, default=0.10)
    parser.add_argument("--log-name", default="p13a_unlabeled_output_distill")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(
        data_yaml=EXP_DIR / "p12_query_set_distill/configs/roadline_no_generic_negatives.yaml",
        cache_root=EXP_DIR / "p12_query_set_distill/cache/dense_teacher_outputs",
        checkpoint=PROJECT_ROOT / "sam3_lightweight_stage3_exp/input/efficientsam3_tinyvit_stage3.pt",
        save=EXP_DIR / "weights/p13a_unlabeled_output_distill.pt", epochs=20, batch_size=4,
    )
    args = parser.parse_args()
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    p0.bind_local_cuda_device(args.accelerator)
    L.seed_everything(args.seed, workers=True)
    dm = CrackYoloSegDataModule(
        data_yaml=args.data_yaml, resolution=args.resolution, prompt_mode="class_name",
        generic_prompt="road marking", batch_size=args.batch_size, num_workers=args.num_workers,
        max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples,
        num_generic_negatives=0,
    )
    dm.setup("fit")
    verify_dense_cache(dm, Path(args.cache_root))
    unlabeled = UnlabeledRoadlineDataset(args.unlabeled_manifest, args.resolution, args.max_unlabeled_samples)
    verify_unlabeled_cache(unlabeled, Path(args.unlabeled_cache_root), list(dm.train_dataset.class_names.values()))
    unlabeled_loader = DataLoader(
        unlabeled, batch_size=args.unlabeled_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_unlabeled, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = P13AUnlabeledModule(
        cache_root=args.cache_root, feature_cache_root=args.feature_cache_root,
        unlabeled_cache_root=args.unlabeled_cache_root, prompts=list(dm.train_dataset.class_names.values()),
        checkpoint=args.checkpoint, resolution=args.resolution, lora_lr=args.lora_lr,
        head_lr=args.head_lr, image_lora_lr=args.image_lora_lr, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        image_lora_rank=args.image_lora_rank, image_lora_alpha=args.image_lora_alpha,
        image_lora_dropout=args.image_lora_dropout, image_lora_stages=tuple(args.image_lora_stages),
        weight_decay=args.weight_decay, kd_weight=args.kd_weight,
        kd_warmup_ratio=args.kd_warmup_ratio, quality_threshold=args.quality_threshold,
        temperature=args.temperature, image_feature_kd_weight=args.image_feature_kd_weight,
        foreground_weight=args.foreground_weight, branch_lr=args.branch_lr, gate_lr=args.gate_lr,
        p5_branch_channels=args.p5_branch_channels, p6_branch_channels=args.p6_branch_channels,
        kernel_size=args.kernel_size, offset_scale=args.offset_scale, p8_operator=args.p8_operator,
        p8_stem_channels=args.p8_stem_channels, p8_line_channels=args.p8_line_channels,
        candidate_set_kd_weight=args.candidate_set_kd_weight, candidate_topk=args.candidate_topk,
        candidate_min_score=args.candidate_min_score, candidate_rank_weight=args.candidate_rank_weight,
        relation_kd_weight=args.relation_kd_weight, relation_iou_scale=args.relation_iou_scale,
        unlabeled_kd_weight=args.unlabeled_kd_weight,
        unlabeled_kd_warmup_ratio=args.unlabeled_kd_warmup_ratio,
    )
    print(f"P13-A从官方TinyViT重新训练；每卡有标签batch={args.batch_size}，无标签batch={args.unlabeled_batch_size}", flush=True)
    callbacks = [p0.SaveBest(args.best_save or p0.default_best_path(args.save))]
    if args.save_every_epoch:
        callbacks.append(p0.SaveEveryEpoch(args.save))
    trainer = L.Trainer(
        default_root_dir=EXP_DIR / "logs" / args.log_name, accelerator=args.accelerator,
        devices=args.devices, strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=1 if args.dry_run else args.epochs, precision=args.precision,
        gradient_clip_val=1.0, callbacks=callbacks, enable_checkpointing=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        limit_train_batches=1 if args.dry_run else 1.0,
        limit_val_batches=1 if args.dry_run else 1.0,
        log_every_n_steps=1 if args.dry_run else 10,
    )
    trainer.fit(model, train_dataloaders={"labeled": dm.train_dataloader(), "unlabeled": unlabeled_loader}, val_dataloaders=dm.val_dataloader())
    if trainer.is_global_zero:
        model.save_lora_checkpoint(args.save)
        print(f"已保存P13-A最后权重：{args.save}", flush=True)


if __name__ == "__main__":
    main()
