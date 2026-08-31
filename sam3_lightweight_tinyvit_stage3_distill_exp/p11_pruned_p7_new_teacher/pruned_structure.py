"""P11：保留P7中分辨率路径，物理移除P7高分辨率路径。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn

P11_DIR = Path(__file__).resolve().parent
EXP_DIR = P11_DIR.parent
for path in (
    EXP_DIR / "p5_dsconv_thin_line", EXP_DIR / "p6_multiscale_dsconv",
    EXP_DIR / "p7_highres_fpn", EXP_DIR / "p8_input_line_branch",
):
    sys.path.insert(0, str(path))

from dsconv_branch import attach_p5_dsconv_branch
from highres_fpn import FPNResidualAdapter
from input_line_branch import attach_p8_input_line_branch
from multiscale_dsconv import attach_p6_stage1_branch

P7_BRANCH_PREFIX = "p7_highres_fpn_adapters."


class MidResolutionFPNAdapter(nn.Module):
    """P7只保留Stage 2方向特征到144×144 FPN的适配器。"""

    def __init__(self, stage2_channels: int) -> None:
        super().__init__()
        self.mid = FPNResidualAdapter(stage2_channels * 2)


@dataclass(frozen=True)
class PrunedP7Attachment:
    mid_fpn_index: int = 1
    fpn_channels: int = 256
    high_branch_enabled: bool = False


def attach_pruned_p7_mid_fpn(detector: nn.Module) -> PrunedP7Attachment:
    """捕获P5的Stage 2方向特征，只注入FPN[1]，不创建high模块。"""
    if not hasattr(detector, "p5_thin_line_branch") or not hasattr(
        detector, "p6_stage1_thin_line_branch"
    ):
        raise RuntimeError("P11要求先挂载P5和P6分支")
    if hasattr(detector, "p7_highres_fpn_adapters"):
        raise RuntimeError("P7适配器已经挂载")

    stage2 = detector.p5_thin_line_branch
    stage2_channels = int(stage2.horizontal.projection.out_channels)
    adapters = MidResolutionFPNAdapter(stage2_channels)
    reference = next(stage2.parameters())
    adapters.to(device=reference.device, dtype=reference.dtype)
    detector.add_module("p7_highres_fpn_adapters", adapters)
    detector._p11_stage2_horizontal = None
    detector._p11_stage2_vertical = None

    def save(name):
        def hook(_module, _inputs, output):
            setattr(detector, name, output)
        return hook

    handles = [
        stage2.horizontal.register_forward_hook(save("_p11_stage2_horizontal")),
        stage2.vertical.register_forward_hook(save("_p11_stage2_vertical")),
    ]
    neck = detector.backbone.vision_backbone
    original_forward = neck.forward

    def forward_with_pruned_p7(_self, tensor_list):
        detector._p11_stage2_horizontal = None
        detector._p11_stage2_vertical = None
        sam3_out, sam3_pos, sam2_out, sam2_pos = original_forward(tensor_list)
        if len(sam3_out) < 3:
            raise RuntimeError(f"P11要求至少3级FPN，实际={len(sam3_out)}")
        horizontal = detector._p11_stage2_horizontal
        vertical = detector._p11_stage2_vertical
        if horizontal is None or vertical is None:
            raise RuntimeError("P11没有捕获到Stage 2横/纵方向特征")
        feature = torch.cat((horizontal, vertical), dim=1)
        residual = detector.p7_highres_fpn_adapters.mid(
            feature, tuple(sam3_out[1].shape[-2:]),
        )
        sam3_out[1] = sam3_out[1] + residual.to(dtype=sam3_out[1].dtype)
        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = MethodType(forward_with_pruned_p7, neck)
    detector._p11_capture_handles = handles
    detector._p11_original_neck_forward = original_forward
    detector.p7_attachment = PrunedP7Attachment()
    return detector.p7_attachment


def attach_pruned_complete_structure(
    detector: nn.Module,
    p5_branch_channels: int = 128,
    p6_branch_channels: int = 64,
    kernel_size: int = 9,
    offset_scale: float = 1.0,
    p8_operator: str = "dsconv",
    p8_stem_channels: int = 16,
    p8_line_channels: int = 16,
) -> None:
    """从官方学生起点挂载P5、P6、P7-mid和P8。"""
    attach_p5_dsconv_branch(
        detector, branch_channels=p5_branch_channels,
        kernel_size=kernel_size, offset_scale=offset_scale,
    )
    attach_p6_stage1_branch(
        detector, branch_channels=p6_branch_channels,
        kernel_size=kernel_size, offset_scale=offset_scale,
    )
    attach_pruned_p7_mid_fpn(detector)
    attach_p8_input_line_branch(
        detector, operator=p8_operator, stem_channels=p8_stem_channels,
        line_channels=p8_line_channels, kernel_size=kernel_size,
        offset_scale=offset_scale,
    )
