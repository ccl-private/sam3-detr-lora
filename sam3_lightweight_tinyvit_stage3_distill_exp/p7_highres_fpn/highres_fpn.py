"""P7：复用P6 DSConv方向特征，直接注入高/中分辨率FPN。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsconv_branch import P5_BRANCH_PREFIX
from multiscale_dsconv import (
    P6_BRANCH_PREFIX,
    attach_multiscale_from_checkpoint,
)


P7_BRANCH_PREFIX = "p7_highres_fpn_adapters."


class FPNResidualAdapter(nn.Module):
    """把冻结DSConv的横纵方向特征投影到一个FPN尺度。"""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(32, out_channels)
        self.activation = nn.GELU()
        self.gate = nn.Parameter(torch.zeros(()))
        self.last_input_shape: tuple[int, ...] | None = None
        self.last_output_shape: tuple[int, ...] | None = None

    def forward(self, feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        self.last_input_shape = tuple(feature.shape)
        output = self.activation(self.norm(self.projection(feature)))
        if output.shape[-2:] != output_size:
            output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        self.last_output_shape = tuple(output.shape)
        return self.gate * output


class HighResolutionFPNAdapters(nn.Module):
    def __init__(self, stage1_channels: int, stage2_channels: int) -> None:
        super().__init__()
        self.high = FPNResidualAdapter(stage1_channels * 2)
        self.mid = FPNResidualAdapter(stage2_channels * 2)


@dataclass(frozen=True)
class P7Attachment:
    high_fpn_index: int
    mid_fpn_index: int
    fpn_channels: int


def attach_p7_highres_fpn(detector: nn.Module) -> P7Attachment:
    """要求P6已挂载；捕获其方向特征并在neck输出处增加零门控残差。"""
    if not hasattr(detector, "p5_thin_line_branch") or not hasattr(
        detector, "p6_stage1_thin_line_branch"
    ):
        raise RuntimeError("P7要求先完整挂载P6的Stage 1/2分支")
    if hasattr(detector, "p7_highres_fpn_adapters"):
        raise RuntimeError("P7高分辨率FPN适配器已经挂载")

    stage1 = detector.p6_stage1_thin_line_branch
    stage2 = detector.p5_thin_line_branch
    stage1_channels = int(stage1.horizontal.projection.out_channels)
    stage2_channels = int(stage2.horizontal.projection.out_channels)
    adapters = HighResolutionFPNAdapters(stage1_channels, stage2_channels)
    reference = next(stage1.parameters())
    adapters.to(device=reference.device, dtype=reference.dtype)
    detector.add_module("p7_highres_fpn_adapters", adapters)

    detector._p7_stage1_horizontal = None
    detector._p7_stage1_vertical = None
    detector._p7_stage2_horizontal = None
    detector._p7_stage2_vertical = None

    def save(name):
        def hook(_module, _inputs, output):
            setattr(detector, name, output)
        return hook

    handles = [
        stage1.horizontal.register_forward_hook(save("_p7_stage1_horizontal")),
        stage1.vertical.register_forward_hook(save("_p7_stage1_vertical")),
        stage2.horizontal.register_forward_hook(save("_p7_stage2_horizontal")),
        stage2.vertical.register_forward_hook(save("_p7_stage2_vertical")),
    ]

    neck = detector.backbone.vision_backbone
    original_forward = neck.forward

    def forward_with_p7(_self, tensor_list):
        detector._p7_stage1_horizontal = None
        detector._p7_stage1_vertical = None
        detector._p7_stage2_horizontal = None
        detector._p7_stage2_vertical = None
        sam3_out, sam3_pos, sam2_out, sam2_pos = original_forward(tensor_list)
        if len(sam3_out) < 3:
            raise RuntimeError(f"P7要求至少3级FPN，实际={len(sam3_out)}")
        captured = (
            detector._p7_stage1_horizontal,
            detector._p7_stage1_vertical,
            detector._p7_stage2_horizontal,
            detector._p7_stage2_vertical,
        )
        if any(value is None for value in captured):
            raise RuntimeError("P7没有捕获到完整的P6横/纵DSConv特征")
        stage1_feature = torch.cat(captured[:2], dim=1)
        stage2_feature = torch.cat(captured[2:], dim=1)
        high = detector.p7_highres_fpn_adapters.high(
            stage1_feature, tuple(sam3_out[0].shape[-2:]),
        )
        mid = detector.p7_highres_fpn_adapters.mid(
            stage2_feature, tuple(sam3_out[1].shape[-2:]),
        )
        sam3_out[0] = sam3_out[0] + high.to(dtype=sam3_out[0].dtype)
        sam3_out[1] = sam3_out[1] + mid.to(dtype=sam3_out[1].dtype)
        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = MethodType(forward_with_p7, neck)
    detector._p7_capture_handles = handles
    detector._p7_original_neck_forward = original_forward
    detector.p7_attachment = P7Attachment(0, 1, 256)
    return detector.p7_attachment


def restore_p6_then_attach_p7(detector: nn.Module, checkpoint_path: Path) -> tuple[dict, P7Attachment]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"无效P6 checkpoint：{checkpoint_path}")
    meta = payload.get("meta", {})
    if not bool(meta.get("p6_multiscale_dsconv", False)):
        raise ValueError(f"P7初始化权重不是P6 checkpoint：{checkpoint_path}")
    attach_multiscale_from_checkpoint(detector, checkpoint_path)
    missing, unexpected = detector.load_state_dict(payload["state_dict"], strict=False)
    missing_old = [
        key for key in missing
        if key.startswith(P5_BRANCH_PREFIX) or key.startswith(P6_BRANCH_PREFIX)
    ]
    unexpected_relevant = [
        key for key in unexpected
        if key.startswith((P5_BRANCH_PREFIX, P6_BRANCH_PREFIX, P7_BRANCH_PREFIX))
    ]
    if missing_old or unexpected_relevant:
        raise RuntimeError(
            f"P6初始化权重不匹配：missing={missing_old}, unexpected={unexpected_relevant}"
        )
    return meta, attach_p7_highres_fpn(detector)


def attach_p7_from_checkpoint(detector: nn.Module, checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    attach_multiscale_from_checkpoint(detector, checkpoint_path)
    attach_p7_highres_fpn(detector)
    return payload.get("meta", {})
