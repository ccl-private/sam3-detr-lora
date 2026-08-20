#!/usr/bin/env python3
"""为 P0 生成 Base DETR 教师的实例匹配软目标缓存。"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from common import cache_file, prompt_key
from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from sam3.train.matcher import BinaryHungarianMatcherV2
from sam3_detr_exp.modular_pipeline import BPE_PATH, build_detector_model
from sam3_detr_exp.utils import (
    PromptTarget,
    build_prompt,
    build_targets,
    load_lora_state,
    make_find_stage,
)
from sam3_detr_exp.utils.detr_lora_data import (
    YoloSegmentationDataset,
    collate_samples,
    parse_data_yaml,
)


@contextmanager
def external_matching(detector):
    original_compute_matching = detector._compute_matching
    original_back_convert = detector.back_convert
    detector._compute_matching = lambda out, targets: None
    detector.back_convert = lambda target: target
    try:
        yield
    finally:
        detector._compute_matching = original_compute_matching
        detector.back_convert = original_back_convert


def negative_target(text: str, resolution: int) -> PromptTarget:
    return PromptTarget(
        text_prompt=text,
        gt_boxes=torch.zeros(0, 4),
        gt_masks=torch.zeros(0, resolution, resolution, dtype=torch.bool),
        class_id=None,
        prompt_kind="generic_negative",
    )


def cache_is_complete(path: Path, expected_prompt_keys: set[str], resolution: int) -> bool:
    """只复用提示词集合完整且分辨率一致的教师缓存。"""
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return (
            payload.get("format_version") == 1
            and payload.get("resolution") == resolution
            and set(payload.get("prompts", {})) == expected_prompt_keys
        )
    except Exception:
        return False


def native_mask_iou(pred_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    if len(pred_logits) == 0:
        return torch.zeros(0, device=pred_logits.device)
    targets = F.interpolate(
        target_masks.unsqueeze(1).float(),
        size=pred_logits.shape[-2:],
        mode="nearest",
    ).squeeze(1).bool()
    pred = pred_logits.sigmoid() > 0.5
    intersection = (pred & targets).flatten(1).sum(1).float()
    union = (pred | targets).flatten(1).sum(1).float().clamp(min=1)
    return intersection / union


def build_prompt_cache(outputs: dict, targets: dict, matcher, prompt_targets) -> list[dict]:
    batch_idx, src_idx, tgt_idx = matcher(
        outputs,
        targets,
        target_is_valid_padded=targets["target_is_valid_padded"],
    )
    result = []
    offsets = []
    offset = 0
    for target in prompt_targets:
        offsets.append(offset)
        offset += len(target.gt_boxes)

    presence = outputs.get("presence_logit_dec")
    for prompt_index, target in enumerate(prompt_targets):
        count = len(target.gt_boxes)
        entry = {
            "presence_logit": (
                presence[prompt_index].detach().float().cpu().reshape(())
                if presence is not None
                else torch.tensor(0.0)
            ),
            "num_targets": count,
            "instances": [None] * count,
        }
        selected = (batch_idx == prompt_index).nonzero().flatten()
        for position in selected.tolist():
            query_index = int(src_idx[position])
            global_target = int(tgt_idx[position])
            local_target = global_target - offsets[prompt_index]
            if not 0 <= local_target < count:
                raise RuntimeError("教师匹配的目标索引越界")
            pred_box = outputs["pred_boxes"][prompt_index, query_index]
            pred_mask = outputs["pred_masks"][prompt_index, query_index]
            target_box = target.gt_boxes[local_target].to(pred_box)
            target_mask = target.gt_masks[local_target : local_target + 1].to(pred_mask.device)
            box_iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_box.unsqueeze(0)),
                    box_cxcywh_to_xyxy(target_box.unsqueeze(0)),
                )
            )[0].clamp(min=0, max=1)
            mask_iou = native_mask_iou(pred_mask.unsqueeze(0), target_mask)[0]
            score = outputs["pred_logits"][prompt_index, query_index].sigmoid().reshape(())
            quality = torch.sqrt((score * mask_iou.clamp(min=0)).clamp(min=0))
            entry["instances"][local_target] = {
                "query_logit": outputs["pred_logits"][prompt_index, query_index]
                .detach().float().cpu(),
                "box": pred_box.detach().float().cpu(),
                "mask_logit": pred_mask.detach().to(dtype=torch.float16).cpu(),
                "score": score.detach().float().cpu(),
                "box_iou": box_iou.detach().float().cpu(),
                "mask_iou": mask_iou.detach().float().cpu(),
                "quality": quality.detach().float().cpu(),
            }
        entry["num_matched"] = sum(instance is not None for instance in entry["instances"])
        entry["num_unmatched"] = count - entry["num_matched"]
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--teacher-lora", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=EXP_DIR / "cache/p0_teacher")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prompt-batch-size", type=int, default=9)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("必须满足 num_shards >= 1 且 0 <= shard_index < num_shards")
    config = parse_data_yaml(args.data_yaml)
    model = build_detector_model(bpe_path=str(BPE_PATH))
    meta, missing, unexpected = load_lora_state(model, args.teacher_lora)
    if missing or unexpected:
        raise RuntimeError(f"教师 LoRA 不匹配：missing={missing}, unexpected={unexpected}")
    model = model.to(args.device).eval()
    matcher = BinaryHungarianMatcherV2(
        focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
        alpha=0.25, gamma=2.0, stable=False,
    )
    generic_candidates = [
        str(text).strip() for text in config.prompt_training.get("generic_negatives", [])
        if str(text).strip()
    ]
    class_names = (
        list(config.class_names.values())
        if isinstance(config.class_names, dict)
        else list(config.class_names)
    )
    expected_prompt_keys = {
        prompt_key(text) for text in [*class_names, *generic_candidates]
    }

    split_dirs = {"train": config.train_dir, "val": config.val_dir}
    for split in args.splits:
        split_dir = split_dirs.get(split)
        if split_dir is None:
            raise ValueError(f"数据配置没有 {split} 目录")
        dataset = YoloSegmentationDataset(
            split_dir=split_dir,
            class_names=config.class_names,
            resolution=args.resolution,
            prompt_mode="class_name",
            generic_prompt="road marking",
            prompt_training=config.prompt_training,
            include_generic_negatives=False,
            max_samples=args.max_samples,
        )
        pending_indices = []
        for index in range(len(dataset)):
            if index % args.num_shards != args.shard_index:
                continue
            image_path = dataset.records[index][0]
            path = cache_file(args.cache_root, split, image_path)
            if args.overwrite or not cache_is_complete(
                path, expected_prompt_keys, args.resolution
            ):
                pending_indices.append(index)
        loader = DataLoader(
            Subset(dataset, pending_indices),
            batch_size=args.image_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_samples,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        completed = 0
        print(
            f"[{split} 分片={args.shard_index}/{args.num_shards}] "
            f"待生成={len(pending_indices)} image_batch={args.image_batch_size} "
            f"workers={args.num_workers}",
            flush=True,
        )
        for samples in loader:
            prompt_targets = []
            prompt_img_ids = []
            prompt_group_ids = []
            group_prompts = [dict() for _ in samples]
            for image_index, sample in enumerate(samples):
                current = list(sample.prompts)
                current.extend(
                    negative_target(text, args.resolution) for text in generic_candidates
                )
                prompt_targets.extend(current)
                prompt_img_ids.extend([image_index] * len(current))
                prompt_group_ids.extend([image_index] * len(current))
            images = torch.stack([sample.image for sample in samples]).to(
                args.device, non_blocking=True
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
            ):
                image_features = model.backbone.forward_image(images)
                for start in range(0, len(prompt_targets), args.prompt_batch_size):
                    chunk = prompt_targets[start : start + args.prompt_batch_size]
                    chunk_img_ids = prompt_img_ids[start : start + args.prompt_batch_size]
                    chunk_group_ids = prompt_group_ids[start : start + args.prompt_batch_size]
                    texts = [target.text_prompt for target in chunk]
                    backbone_out = dict(image_features)
                    backbone_out.update(
                        model.backbone.forward_text(texts, device=torch.device(args.device))
                    )
                    find_input = make_find_stage(
                        torch.tensor(chunk_img_ids, dtype=torch.long), torch.device(args.device)
                    )
                    geometric_prompt = build_prompt(model, len(texts), torch.device(args.device))
                    with external_matching(model):
                        outputs = model.forward_grounding(
                            backbone_out=backbone_out,
                            find_input=find_input,
                            find_target=None,
                            geometric_prompt=geometric_prompt,
                        )
                    targets = build_targets(chunk, torch.device(args.device))
                    entries = build_prompt_cache(outputs, targets, matcher, chunk)
                    for target, group_id, entry in zip(chunk, chunk_group_ids, entries):
                        group_prompts[group_id][prompt_key(target.text_prompt)] = entry
            for sample, prompts in zip(samples, group_prompts):
                output_path = cache_file(args.cache_root, split, sample.image_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "format_version": 1,
                        "image_path": str(sample.image_path.expanduser().resolve()),
                        "split": split,
                        "resolution": args.resolution,
                        "teacher_lora": str(args.teacher_lora.resolve()),
                        "teacher_meta": meta,
                        "prompts": prompts,
                    },
                    output_path,
                )
                completed += 1
            if completed % max(1, args.log_every) < len(samples):
                print(
                    f"[{split} 分片={args.shard_index}/{args.num_shards}] "
                    f"本次={completed}/{len(pending_indices)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
