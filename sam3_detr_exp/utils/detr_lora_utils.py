from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from sam3.model.data_misc import FindStage
from sam3.train.matcher import BinaryHungarianMatcherV2
from sam3_detr_exp.modular_pipeline import BPE_PATH, WEIGHTS_DIR, build_detector_model
from sam3_detr_exp.utils.detr_lora_data import Sample

ROOT = Path(__file__).resolve().parent.parent
LORA_STATE_PREFIXES = ("dot_prod_scoring.", "segmentation_head.")
EXPECTED_MODULES = [
    "vision_backbone",
    "text_encoder",
    "transformer_encoder",
    "transformer_decoder",
    "segmentation_head",
    "geometry_encoder",
    "dot_product_scoring",
]


def assert_modular_weights_exist() -> None:
    missing = [name for name in EXPECTED_MODULES if not (WEIGHTS_DIR / f"{name}.pt").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing modular weight files: "
            + ", ".join(missing)
            + f". Run {ROOT / 'run_video_det_modular.py'} first."
        )


class LoRAParametrization(nn.Module):
    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        delta = self.lora_b @ self.lora_a
        return base_weight + delta * self.scale


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _register_weight_parametrization(
    module: nn.Module, parameter_name: str, parametrization: nn.Module
) -> None:
    import torch.nn.utils.parametrize as parametrize

    existing = getattr(module, "parametrizations", None)
    if existing is not None and hasattr(existing, parameter_name):
        return
    parametrize.register_parametrization(module, parameter_name, parametrization)


def attach_lora_to_parametrizable_modules(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    include_encoder: bool = True,
    include_decoder: bool = True,
    include_ffn: bool = True,
) -> list[str]:
    attached: list[str] = []
    for module_name, module in model.transformer.named_modules():
        is_encoder = module_name.startswith("encoder.layers.")
        is_decoder = module_name.startswith("decoder.layers.")
        if not ((include_encoder and is_encoder) or (include_decoder and is_decoder)):
            continue

        if isinstance(module, nn.MultiheadAttention):
            _register_weight_parametrization(
                module,
                "in_proj_weight",
                LoRAParametrization(
                    module.in_proj_weight.shape[0],
                    module.in_proj_weight.shape[1],
                    rank,
                    alpha,
                    dropout,
                    device=module.in_proj_weight.device,
                    dtype=module.in_proj_weight.dtype,
                ),
            )
            attached.append(f"{module_name}.in_proj_weight")
            _register_weight_parametrization(
                module.out_proj,
                "weight",
                LoRAParametrization(
                    module.out_proj.weight.shape[0],
                    module.out_proj.weight.shape[1],
                    rank,
                    alpha,
                    dropout,
                    device=module.out_proj.weight.device,
                    dtype=module.out_proj.weight.dtype,
                ),
            )
            attached.append(f"{module_name}.out_proj.weight")
        elif (
            include_ffn
            and isinstance(module, nn.Linear)
            and module_name.endswith(("linear1", "linear2"))
        ):
            _register_weight_parametrization(
                module,
                "weight",
                LoRAParametrization(
                    module.weight.shape[0],
                    module.weight.shape[1],
                    rank,
                    alpha,
                    dropout,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                ),
            )
            attached.append(f"{module_name}.weight")
    return attached


def collect_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def build_trainable_detector(
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    decoder_only: bool,
    attn_only: bool,
    train_dot_score: bool,
    train_seg_head: bool,
) -> tuple[nn.Module, list[str]]:
    model = build_detector_model(bpe_path=str(BPE_PATH))
    freeze_module(model)

    if train_dot_score:
        for param in model.dot_prod_scoring.parameters():
            param.requires_grad = True
    if train_seg_head:
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

    attached = attach_lora_to_parametrizable_modules(
        model=model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        include_encoder=not decoder_only,
        include_decoder=True,
        include_ffn=not attn_only,
    )
    return model, attached


def set_frozen_module_modes(
    model: nn.Module, train_dot_score: bool, train_seg_head: bool
) -> None:
    model.eval()
    model.transformer.train()
    model.backbone.eval()
    model.geometry_encoder.eval()
    if train_dot_score:
        model.dot_prod_scoring.train()
    else:
        model.dot_prod_scoring.eval()
    if train_seg_head:
        model.segmentation_head.train()
    else:
        model.segmentation_head.eval()


def make_find_stage(batch_size: int, device: torch.device) -> FindStage:
    return FindStage(
        img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        text_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        input_boxes=torch.zeros(batch_size, 0, 4, device=device, dtype=torch.float32),
        input_boxes_mask=torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
        input_boxes_label=torch.zeros(batch_size, 0, device=device, dtype=torch.long),
        input_points=torch.empty(batch_size, 0, 2, device=device, dtype=torch.float32),
        input_points_mask=torch.empty(batch_size, 0, device=device, dtype=torch.bool),
        object_ids=[[] for _ in range(batch_size)],
    )


def build_prompt(model: nn.Module, samples: list[Sample], device: torch.device):
    prompt = model._get_dummy_prompt(num_prompts=len(samples))
    return prompt


def build_targets(samples: list[Sample], device: torch.device) -> dict[str, torch.Tensor]:
    num_boxes = torch.tensor(
        [len(sample.gt_boxes) for sample in samples], dtype=torch.long, device=device
    )
    boxes = torch.cat([sample.gt_boxes for sample in samples], dim=0).to(device)
    max_boxes = int(num_boxes.max().item())
    boxes_padded = torch.zeros(
        len(samples), max_boxes, 4, dtype=torch.float32, device=device
    )
    masks = []
    valid_masks = []
    for idx, sample in enumerate(samples):
        count = len(sample.gt_boxes)
        boxes_padded[idx, :count] = sample.gt_boxes.to(device)
        masks.append(sample.gt_masks.to(device))
        valid_masks.append(torch.ones(count, dtype=torch.bool, device=device))
    masks_tensor = torch.cat(masks, dim=0)
    target_is_valid_padded = torch.zeros(
        len(samples), max_boxes, dtype=torch.bool, device=device
    )
    for idx, sample in enumerate(samples):
        target_is_valid_padded[idx, : len(sample.gt_boxes)] = True
    return {
        "boxes": boxes,
        "boxes_padded": boxes_padded,
        "num_boxes": num_boxes,
        "masks": masks_tensor,
        "is_valid_mask": torch.cat(valid_masks, dim=0),
        "target_is_valid_padded": target_is_valid_padded,
    }


def compute_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    matcher: BinaryHungarianMatcherV2,
    resolution: int,
    mask_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    batch_idx, src_idx, tgt_idx = matcher(
        outputs,
        targets,
        target_is_valid_padded=targets["target_is_valid_padded"],
    )
    tgt_idx = tgt_idx.view(-1)

    logits = outputs["pred_logits"].squeeze(-1)
    target_logits = torch.zeros_like(logits)
    if len(src_idx) > 0:
        target_logits[batch_idx, src_idx] = 1.0
    loss_cls = F.binary_cross_entropy_with_logits(logits, target_logits)

    if len(src_idx) == 0:
        loss_box = outputs["pred_boxes"].sum() * 0.0
        loss_giou = outputs["pred_boxes"].sum() * 0.0
        loss_mask = outputs["pred_masks"].sum() * 0.0
    else:
        matched_pred_boxes = outputs["pred_boxes"][batch_idx, src_idx]
        matched_tgt_boxes = targets["boxes"][tgt_idx]
        loss_box = F.l1_loss(matched_pred_boxes, matched_tgt_boxes)

        pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt_boxes)
        loss_giou = (
            1.0 - torch.diag(generalized_box_iou(pred_xyxy, tgt_xyxy))
        ).mean()

        matched_pred_masks = outputs["pred_masks"][batch_idx, src_idx]
        matched_pred_masks = F.interpolate(
            matched_pred_masks.unsqueeze(1),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        matched_tgt_masks = targets["masks"][tgt_idx].float()
        loss_mask = F.binary_cross_entropy_with_logits(
            matched_pred_masks, matched_tgt_masks
        )

    loss = loss_cls + 5.0 * loss_box + 2.0 * loss_giou + mask_weight * loss_mask
    metrics: dict[str, torch.Tensor | int] = {
        "loss": loss.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_box": loss_box.detach(),
        "loss_giou": loss_giou.detach(),
        "loss_mask": loss_mask.detach(),
        "num_matches": int(len(src_idx)),
    }
    return loss, metrics


def _lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "parametrizations" in name or name.startswith(LORA_STATE_PREFIXES)
    }


def save_lora_state(model: nn.Module, output_path: Path, meta: dict) -> None:
    payload = {"meta": meta, "state_dict": _lora_state_dict(model)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_lora_state(model: nn.Module, checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid LoRA checkpoint format: {checkpoint_path}")

    meta = payload.get("meta", {})
    attach_lora_to_parametrizable_modules(
        model=model,
        rank=int(meta.get("lora_rank", 8)),
        alpha=float(meta.get("lora_alpha", 16.0)),
        dropout=float(meta.get("lora_dropout", 0.05)),
        include_encoder=not bool(meta.get("decoder_only", False)),
        include_decoder=True,
        include_ffn=not bool(meta.get("attn_only", False)),
    )
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    missing = [
        key
        for key in missing
        if "parametrizations" in key or key.startswith(LORA_STATE_PREFIXES)
    ]
    unexpected = [
        key
        for key in unexpected
        if "parametrizations" in key or key.startswith(LORA_STATE_PREFIXES)
    ]
    return meta, missing, unexpected
