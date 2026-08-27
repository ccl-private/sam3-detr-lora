"""在官方TinyViT Stage-3上从零挂载P5～P8完整细线结构。"""

from __future__ import annotations

import torch.nn as nn

from dsconv_branch import attach_p5_dsconv_branch
from highres_fpn import attach_p7_highres_fpn
from input_line_branch import attach_p8_input_line_branch
from multiscale_dsconv import attach_p6_stage1_branch


def attach_complete_p8_structure(
    detector: nn.Module,
    p5_branch_channels: int = 128,
    p6_branch_channels: int = 64,
    kernel_size: int = 9,
    offset_scale: float = 1.0,
    p8_operator: str = "dsconv",
    p8_stem_channels: int = 16,
    p8_line_channels: int = 16,
) -> None:
    """不加载历史学生权重，按P5→P6→P7→P8顺序挂载全部结构。"""
    attach_p5_dsconv_branch(
        detector, branch_channels=p5_branch_channels,
        kernel_size=kernel_size, offset_scale=offset_scale,
    )
    attach_p6_stage1_branch(
        detector, branch_channels=p6_branch_channels,
        kernel_size=kernel_size, offset_scale=offset_scale,
    )
    attach_p7_highres_fpn(detector)
    attach_p8_input_line_branch(
        detector, operator=p8_operator, stem_channels=p8_stem_channels,
        line_channels=p8_line_channels, kernel_size=kernel_size,
        offset_scale=offset_scale,
    )
