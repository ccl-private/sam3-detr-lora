#!/usr/bin/env python3
"""缓存P13-A无标签图片的新Base教师高置信输出。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

P13_DIR = Path(__file__).resolve().parent
EXP_DIR = P13_DIR.parent
PROJECT_ROOT = EXP_DIR.parent
for path in (PROJECT_ROOT, EXP_DIR, P13_DIR):
    sys.path.insert(0, str(path))

from common import cache_file, prompt_key
from sam3_detr_exp.modular_pipeline import BPE_PATH, build_detector_model
from sam3_detr_exp.model.detr_lora_module import _external_matching, _reset_stale_decoder_caches
from sam3_detr_exp.utils import build_prompt, load_lora_state, make_find_stage
from sam3_detr_exp.utils.detr_lora_data import parse_data_yaml
from unlabeled_data import UnlabeledRoadlineDataset, collate_unlabeled


def cache_complete(path: Path, prompts: list[str], resolution: int, threshold: float, mask_nms_iou: float) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return (
            payload.get("format_version") == 1
            and payload.get("resolution") == resolution
            and abs(float(payload.get("threshold", -1)) - threshold) < 1e-9
            and abs(float(payload.get("mask_nms_iou", -1)) - mask_nms_iou) < 1e-9
            and set(payload.get("prompts", {})) == {prompt_key(text) for text in prompts}
        )
    except Exception:
        return False


def mask_nms(indices: torch.Tensor, masks: torch.Tensor, threshold: float) -> list[int]:
    kept = []
    binary = masks[indices].sigmoid() > 0.5
    for local_index, query_index in enumerate(indices.tolist()):
        candidate = binary[local_index]
        duplicate = False
        for kept_local in kept:
            reference = binary[kept_local]
            union = (candidate | reference).sum().clamp(min=1)
            if float((candidate & reference).sum() / union) > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(local_index)
    return [int(indices[local].item()) for local in kept]


def prompt_entry(outputs: dict, index: int, threshold: float, topk: int, mask_nms_iou: float) -> dict:
    logits = outputs["pred_logits"][index].float().reshape(-1)
    boxes = outputs["pred_boxes"][index].float().reshape(-1, 4)
    masks = outputs["pred_masks"][index].float()
    presence = outputs.get("presence_logit_dec")
    presence_logit = presence[index].float().reshape(()) if presence is not None else logits.new_zeros(())
    combined = logits.sigmoid() * presence_logit.sigmoid()
    chosen = torch.nonzero(combined >= threshold, as_tuple=False).flatten()
    chosen = chosen[torch.argsort(combined[chosen], descending=True)[:topk]]
    chosen = mask_nms(chosen, masks, mask_nms_iou)
    instances = []
    for query_index in chosen:
        instances.append({
            "query_logit": logits[query_index].detach().cpu(),
            "box": boxes[query_index].detach().cpu(),
            "mask_logit": masks[query_index].detach().to(dtype=torch.float16).cpu(),
            "combined_score": combined[query_index].detach().cpu(),
        })
    return {
        "presence_logit": presence_logit.detach().cpu(),
        "num_instances": len(instances),
        "instances": instances,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--teacher-lora", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=P13_DIR / "cache/unlabeled_teacher_outputs")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--mask-nms-iou", type=float, default=0.85)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("分片参数不合法")

    config = parse_data_yaml(args.data_yaml)
    prompts = list(config.class_names.values()) if isinstance(config.class_names, dict) else list(config.class_names)
    dataset = UnlabeledRoadlineDataset(args.manifest, args.resolution, args.max_samples)
    pending = []
    for index, row in enumerate(dataset.records):
        if index % args.num_shards != args.shard_index:
            continue
        path = cache_file(args.cache_root, "train", row["image_path"])
        if args.overwrite or not cache_complete(path, prompts, args.resolution, args.threshold, args.mask_nms_iou):
            pending.append(index)
    loader = DataLoader(
        Subset(dataset, pending), batch_size=args.image_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_unlabeled, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_detector_model(bpe_path=str(BPE_PATH))
    meta, missing, unexpected = load_lora_state(model, args.teacher_lora)
    if missing or unexpected:
        raise RuntimeError(f"教师权重不匹配：missing={missing}, unexpected={unexpected}")
    device = torch.device(args.device)
    model = model.to(device).eval()
    print(f"分片={args.shard_index}/{args.num_shards}，待缓存={len(pending)}，提示词={len(prompts)}", flush=True)

    completed = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        paths = batch["image_paths"]
        text = prompts * len(paths)
        image_ids = [image_index for image_index in range(len(paths)) for _ in prompts]
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            _reset_stale_decoder_caches(model, device)
            backbone_out = model.backbone.forward_image(images)
            backbone_out.update(model.backbone.forward_text(text, device=device))
            with _external_matching(model):
                outputs = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=make_find_stage(torch.tensor(image_ids), device),
                    find_target=None,
                    geometric_prompt=build_prompt(model, len(text), device),
                )
        for image_index, image_path in enumerate(paths):
            entries = {}
            for prompt_index, prompt in enumerate(prompts):
                output_index = image_index * len(prompts) + prompt_index
                entries[prompt_key(prompt)] = prompt_entry(outputs, output_index, args.threshold, args.topk, args.mask_nms_iou)
            output_path = cache_file(args.cache_root, "train", image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format_version": 1,
                "image_path": str(Path(image_path).expanduser().resolve()),
                "resolution": args.resolution,
                "threshold": args.threshold,
                "topk": args.topk,
                "mask_nms_iou": args.mask_nms_iou,
                "teacher_lora": str(args.teacher_lora.expanduser().resolve()),
                "teacher_meta": meta,
                "prompts": entries,
            }, output_path)
            completed += 1
        print(f"分片={args.shard_index}，本次完成={completed}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
