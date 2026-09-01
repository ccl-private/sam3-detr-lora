#!/usr/bin/env python3
"""P12：道路标线候选集合与跨提示软关系蒸馏。"""
from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

P12_DIR = Path(__file__).resolve().parent
EXP_DIR = P12_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
STAGE3_DIR = PROJECT_ROOT / "sam3_lightweight_stage3_exp"
for path in (PROJECT_ROOT, STAGE3_DIR, EXP_DIR, EXP_DIR / "p1_image_feature",
             EXP_DIR / "p5_dsconv_thin_line", EXP_DIR / "p6_multiscale_dsconv",
             EXP_DIR / "p7_highres_fpn", EXP_DIR / "p8_input_line_branch",
             EXP_DIR / "p9_fresh_p8_new_teacher", P12_DIR):
    sys.path.insert(0, str(path))

from bootstrap import activate_efficientsam3
activate_efficientsam3()

import train_p0_image_lora as p0
from common import cache_file, load_cache, prompt_key
from p9_fresh_p8_new_teacher.train_p9_fresh_p8 import P9FreshP8Module
from sam3.model.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from sam3_detr_exp.model.detr_lora_module import _external_matching, _move_tensors_to_device, _reset_stale_decoder_caches
from sam3_detr_exp.utils import CrackYoloSegDataModule, build_prompt, build_targets, compute_sam3_losses, make_find_stage


class P12QuerySetModule(P9FreshP8Module):
    """P9同条件基线加道路标线候选集合和跨提示软关系KD。"""
    def __init__(self, *args, candidate_set_kd_weight=0.20, candidate_topk=50,
                 candidate_min_score=0.05, candidate_rank_weight=0.10,
                 relation_kd_weight=0.10, relation_iou_scale=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        if candidate_topk < 1:
            raise ValueError("candidate_topk必须大于0")
        for key, value in {
            "candidate_set_kd_weight": candidate_set_kd_weight,
            "candidate_topk": candidate_topk,
            "candidate_min_score": candidate_min_score,
            "candidate_rank_weight": candidate_rank_weight,
            "relation_kd_weight": relation_kd_weight,
            "relation_iou_scale": relation_iou_scale,
        }.items():
            self.hparams[key] = value
        self._p12_payloads, self._p12_prompt_indices, self._p12_batch = [], [], None

    @staticmethod
    def _dense(entry, device):
        dense = entry.get("dense_queries")
        if dense is None:
            raise RuntimeError("P12需要format_version=2的dense教师缓存；请先执行缓存脚本")
        logits = dense["logits"].to(device=device, dtype=torch.float32).reshape(-1)
        boxes = dense["boxes"].to(device=device, dtype=torch.float32).reshape(-1, 4)
        if len(logits) != len(boxes):
            raise RuntimeError("P12教师dense缓存的logit和box数量不一致")
        return logits, boxes

    def _candidate_set_kd(self, outputs, prompt_caches):
        """高分候选集合匹配；全部Q个候选只蒸馏无序置信度分布。"""
        device, temp = outputs["pred_logits"].device, float(self.hparams.temperature)
        zero = outputs["pred_logits"].sum() * 0.0
        top_losses, rank_losses, match_count = [], [], 0
        for p, entry in enumerate(prompt_caches):
            tl, tb = self._dense(entry, device)
            sl = outputs["pred_logits"][p].float().reshape(-1)
            sb = outputs["pred_boxes"][p].float().reshape(-1, 4)
            if len(tl) != len(sl):
                raise RuntimeError(f"P12 query数不一致：teacher={len(tl)} student={len(sl)}")
            # 不按query编号对齐：全部Q个候选的排序分布用于校准强弱候选比例。
            rank_losses.append(F.binary_cross_entropy_with_logits(
                torch.sort(sl)[0] / temp, (torch.sort(tl)[0] / temp).sigmoid()) * temp**2)
            scores = tl.sigmoid()
            chosen = torch.nonzero(scores >= float(self.hparams.candidate_min_score), as_tuple=False).flatten()
            if not len(chosen):
                continue
            chosen = chosen[torch.argsort(scores[chosen], descending=True)[:int(self.hparams.candidate_topk)]]
            tbox = tb[chosen]
            giou_matrix = generalized_box_iou(box_cxcywh_to_xyxy(tbox), box_cxcywh_to_xyxy(sb))
            cost = (torch.cdist(tbox, sb, p=1) + 2.0 * (1.0 - giou_matrix)).detach().cpu().numpy()
            rows, cols = linear_sum_assignment(cost)
            if not len(rows):
                continue
            tids, sids = chosen[torch.as_tensor(rows, device=device)], torch.as_tensor(cols, device=device)
            weight = scores[tids].sqrt(); weight = weight / weight.sum().clamp(min=1e-6)
            cls = F.binary_cross_entropy_with_logits(sl[sids] / temp, (tl[tids] / temp).sigmoid(), reduction="none") * temp**2
            l1 = F.l1_loss(sb[sids], tb[tids], reduction="none").mean(1)
            giou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(sb[sids]), box_cxcywh_to_xyxy(tb[tids])))
            top_losses.append((weight * (cls + l1 + 2.0 * (1.0 - giou))).sum())
            match_count += len(tids)
        top = torch.stack(top_losses).mean() if top_losses else zero
        rank = torch.stack(rank_losses).mean() if rank_losses else zero
        return top + float(self.hparams.candidate_rank_weight) * rank, {
            "kd_candidate_set": top.detach(), "kd_candidate_rank": rank.detach(),
            "kd_candidate_matches": torch.tensor(float(match_count), device=device),
        }

    def _proposal_response(self, logits, boxes, anchor):
        iou, _ = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(anchor.unsqueeze(0)))
        spatial = float(self.hparams.relation_iou_scale) * torch.log(iou[:, 0].detach().clamp(min=1e-4))
        return torch.logsumexp(logits.reshape(-1) + spatial, dim=0)

    def _relation_kd(self, outputs):
        """在每个GT实例锚点上构造7提示教师/学生软响应向量。"""
        device, temp = outputs["pred_logits"].device, float(self.hparams.temperature)
        zero, losses, count = outputs["pred_logits"].sum() * 0.0, [], 0
        for image_i, sample in enumerate(self._p12_batch):
            payload, indices = self._p12_payloads[image_i], self._p12_prompt_indices[image_i]
            for anchor_target in sample.prompts:
                for anchor in anchor_target.gt_boxes.to(device):
                    tv, sv = [], []
                    for target, output_i in zip(sample.prompts, indices):
                        entry = payload["prompts"].get(prompt_key(target.text_prompt))
                        if entry is None:
                            raise KeyError(f"P12教师缓存缺少关系提示：{target.text_prompt}")
                        tl, tb = self._dense(entry, device)
                        tv.append(self._proposal_response(tl, tb, anchor))
                        sv.append(self._proposal_response(outputs["pred_logits"][output_i].float(), outputs["pred_boxes"][output_i].float(), anchor))
                    losses.append(F.kl_div(F.log_softmax(torch.stack(sv) / temp, dim=0), F.softmax(torch.stack(tv) / temp, dim=0), reduction="sum") * temp**2)
                    count += 1
        loss = torch.stack(losses).mean() if losses else zero
        return loss, {"kd_cross_prompt_relation": loss.detach(), "kd_relation_anchors": torch.tensor(float(count), device=device)}

    def compute_kd(self, outputs, targets, prompt_targets, prompt_caches):
        base, metrics = super().compute_kd(outputs, targets, prompt_targets, prompt_caches)
        candidate, cm = self._candidate_set_kd(outputs, prompt_caches)
        relation, rm = self._relation_kd(outputs)
        metrics.update(cm); metrics.update(rm)
        metrics["kd_p12_candidate_weight"] = torch.tensor(float(self.hparams.candidate_set_kd_weight), device=self.device)
        metrics["kd_p12_relation_weight"] = torch.tensor(float(self.hparams.relation_kd_weight), device=self.device)
        return base + float(self.hparams.candidate_set_kd_weight) * candidate + float(self.hparams.relation_kd_weight) * relation, metrics

    def _p12_shared_step(self, batch, stage):
        if stage == "train": self._set_trainable_modes()
        images = torch.stack([sample.image for sample in batch]).to(self.device, non_blocking=True)
        prompt_targets, prompt_img_ids, prompt_caches = [], [], []
        self._p12_payloads, self._p12_prompt_indices = [], []
        split = "train" if stage == "train" else "val"
        for image_i, sample in enumerate(batch):
            payload = load_cache(Path(self.hparams.cache_root), split, sample.image_path)
            if not payload.get("dense_queries"):
                raise RuntimeError("P12需要dense_queries=true的教师缓存")
            indices = []
            for target in sample.prompts:
                key = prompt_key(target.text_prompt)
                if key not in payload["prompts"]: raise KeyError(f"教师缓存缺少提示：{target.text_prompt}")
                indices.append(len(prompt_targets)); prompt_targets.append(target); prompt_img_ids.append(image_i); prompt_caches.append(payload["prompts"][key])
            self._p12_payloads.append(payload); self._p12_prompt_indices.append(indices)
        self._p12_batch = batch
        texts = [target.text_prompt for target in prompt_targets]
        _reset_stale_decoder_caches(self.detector, self.device)
        backbone_out = self.detector.backbone.forward_image(images)
        with torch.no_grad(): backbone_out.update(self.detector.backbone.forward_text(texts, device=self.device))
        backbone_out = _move_tensors_to_device(backbone_out, self.device)
        with _external_matching(self.detector):
            outputs = self.detector.forward_grounding(backbone_out=backbone_out, find_input=make_find_stage(torch.tensor(prompt_img_ids), self.device), find_target=None, geometric_prompt=build_prompt(self.detector, len(prompt_targets), self.device))
        targets = build_targets(prompt_targets, self.device)
        supervised, sm = compute_sam3_losses(outputs, targets, matcher=self.matcher, o2m_matcher=self.o2m_matcher, loss_fns=self.sam3_loss_fns)
        kd, km = self.compute_kd(outputs, targets, prompt_targets, prompt_caches)
        total, scale, batch_size = supervised + self.kd_scale() * kd, self.kd_scale(), len(batch)
        self.log(f"{stage}/loss", total, prog_bar=True, batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/supervised", supervised.detach(), batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/kd", kd.detach(), batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/kd_scale", scale, batch_size=batch_size, sync_dist=stage == "val")
        self.log(f"{stage}/num_matches", float(sm["num_matches"]), batch_size=batch_size, sync_dist=stage == "val")
        for key, value in km.items(): self.log(f"{stage}/{key}", value, batch_size=batch_size, sync_dist=stage == "val")
        return total

    def _shared_step(self, batch, stage):
        split = "train" if stage == "train" else "val"; self._image_feature_kd = None
        with self._capture_image_feature_loss(batch, split): return self._p12_shared_step(batch, stage)

    def save_lora_checkpoint(self, path):
        super().save_lora_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["meta"].update({"experiment": "P12 P9 baseline plus roadline query-set and cross-prompt KD", "p12_candidate_set_kd": True, "p12_candidate_set_kd_weight": float(self.hparams.candidate_set_kd_weight), "p12_candidate_topk": int(self.hparams.candidate_topk), "p12_candidate_min_score": float(self.hparams.candidate_min_score), "p12_candidate_rank_weight": float(self.hparams.candidate_rank_weight), "p12_cross_prompt_relation_kd": True, "p12_relation_kd_weight": float(self.hparams.relation_kd_weight), "p12_relation_iou_scale": float(self.hparams.relation_iou_scale)})
        torch.save(payload, path)


def verify_dense_cache(datamodule, cache_root: Path) -> None:
    """在DDP启动前确认训练和验证样本均有P12格式教师缓存。"""
    missing = []
    for split, dataset in (("train", datamodule.train_dataset), ("val", datamodule.val_dataset)):
        if dataset is None:
            raise RuntimeError(f"P12数据集未初始化：{split}")
        for record in dataset.records:
            image_path = record[0]
            path = cache_file(cache_root, split, image_path)
            if not path.exists():
                missing.append((split, path, image_path))
    if missing:
        split, path, image_path = missing[0]
        raise RuntimeError(
            f"P12 dense教师缓存未完成：缺少{len(missing)}份；"
            f"首个为split={split}，缓存={path}，图片={image_path}。"
            "请先完成cache_dense_teacher_queries_4gpu.sh后再启动训练。"
        )


def main():
    parser = p0.build_parser(); parser.description = __doc__
    parser.add_argument("--feature-cache-root", type=Path, required=True); parser.add_argument("--image-feature-kd-weight", type=float, default=1.0); parser.add_argument("--foreground-weight", type=float, default=4.0); parser.add_argument("--image-lora-stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--branch-lr", type=float, default=1e-4); parser.add_argument("--gate-lr", type=float, default=1e-3); parser.add_argument("--p5-branch-channels", type=int, default=128); parser.add_argument("--p6-branch-channels", type=int, default=64); parser.add_argument("--kernel-size", type=int, default=9); parser.add_argument("--offset-scale", type=float, default=1.0); parser.add_argument("--p8-operator", choices=("dsconv", "strip_conv"), default="dsconv"); parser.add_argument("--p8-stem-channels", type=int, default=16); parser.add_argument("--p8-line-channels", type=int, default=16)
    parser.add_argument("--candidate-set-kd-weight", type=float, default=0.20); parser.add_argument("--candidate-topk", type=int, default=50); parser.add_argument("--candidate-min-score", type=float, default=0.05); parser.add_argument("--candidate-rank-weight", type=float, default=0.10); parser.add_argument("--relation-kd-weight", type=float, default=0.10); parser.add_argument("--relation-iou-scale", type=float, default=4.0); parser.add_argument("--log-name", default="p12_query_set_distill"); parser.add_argument("--save-every-epoch", action="store_true")
    parser.set_defaults(data_yaml=P12_DIR / "configs/roadline_no_generic_negatives.yaml", cache_root=P12_DIR / "cache/dense_teacher_outputs", save=EXP_DIR / "weights/p12_query_set_distill.pt", epochs=20, batch_size=4)
    args = parser.parse_args()
    if args.accelerator == "gpu" and not torch.cuda.is_available(): raise RuntimeError("CUDA不可用")
    p0.bind_local_cuda_device(args.accelerator); L.seed_everything(args.seed, workers=True)
    dm = CrackYoloSegDataModule(data_yaml=args.data_yaml, resolution=args.resolution, prompt_mode="class_name", generic_prompt="road marking", batch_size=args.batch_size, num_workers=args.num_workers, max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples, num_generic_negatives=0); dm.setup("fit")
    verify_dense_cache(dm, Path(args.cache_root))
    model = P12QuerySetModule(cache_root=args.cache_root, feature_cache_root=args.feature_cache_root, checkpoint=args.checkpoint, resolution=args.resolution, lora_lr=args.lora_lr, head_lr=args.head_lr, image_lora_lr=args.image_lora_lr, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, image_lora_rank=args.image_lora_rank, image_lora_alpha=args.image_lora_alpha, image_lora_dropout=args.image_lora_dropout, image_lora_stages=tuple(args.image_lora_stages), weight_decay=args.weight_decay, kd_weight=args.kd_weight, kd_warmup_ratio=args.kd_warmup_ratio, quality_threshold=args.quality_threshold, temperature=args.temperature, image_feature_kd_weight=args.image_feature_kd_weight, foreground_weight=args.foreground_weight, branch_lr=args.branch_lr, gate_lr=args.gate_lr, p5_branch_channels=args.p5_branch_channels, p6_branch_channels=args.p6_branch_channels, kernel_size=args.kernel_size, offset_scale=args.offset_scale, p8_operator=args.p8_operator, p8_stem_channels=args.p8_stem_channels, p8_line_channels=args.p8_line_channels, candidate_set_kd_weight=args.candidate_set_kd_weight, candidate_topk=args.candidate_topk, candidate_min_score=args.candidate_min_score, candidate_rank_weight=args.candidate_rank_weight, relation_kd_weight=args.relation_kd_weight, relation_iou_scale=args.relation_iou_scale)
    print(f"P12起点=官方TinyViT Stage-3，域外负提示=0，总参数={sum(p.numel() for p in model.parameters()):,}，可训练参数={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    callbacks = [p0.SaveBest(args.best_save or p0.default_best_path(args.save))]
    if args.save_every_epoch: callbacks.append(p0.SaveEveryEpoch(args.save))
    trainer = L.Trainer(default_root_dir=EXP_DIR / "logs" / args.log_name, accelerator=args.accelerator, devices=args.devices, strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto", max_epochs=1 if args.dry_run else args.epochs, precision=args.precision, gradient_clip_val=1.0, callbacks=callbacks, enable_checkpointing=False, enable_model_summary=False, num_sanity_val_steps=0, limit_train_batches=1 if args.dry_run else 1.0, limit_val_batches=1 if args.dry_run else 1.0, log_every_n_steps=1 if args.dry_run else 10)
    trainer.fit(model, datamodule=dm)
    if trainer.is_global_zero: model.save_lora_checkpoint(args.save); print(f"已保存P12最后权重：{args.save}", flush=True)

if __name__ == "__main__": main()
