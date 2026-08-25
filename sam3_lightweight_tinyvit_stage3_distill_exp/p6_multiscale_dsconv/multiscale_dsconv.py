"""P6 Stage 1 + Stage 2双尺度DSConv接入与checkpoint加载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn

from dsconv_branch import P5_BRANCH_PREFIX, ThinLineDSConvBranch, attach_p5_dsconv_branch


P6_BRANCH_PREFIX = "p6_stage1_thin_line_branch."


@dataclass(frozen=True)
class P6Attachment:
    stage1_channels: int
    stage1_resolution: tuple[int, int]
    output_channels: int


def attach_p6_stage1_branch(
    detector: nn.Module,
    branch_channels: int = 64,
    kernel_size: int = 9,
    offset_scale: float = 1.0,
) -> P6Attachment:
    """在已挂载P5 Stage 2分支的模型上增加零门控Stage 1分支。"""
    if not hasattr(detector, "p5_thin_line_branch"):
        raise RuntimeError("P6要求先挂载并恢复P5 Stage 2分支")
    if hasattr(detector, "p6_stage1_thin_line_branch"):
        raise RuntimeError("P6 Stage 1 DSConv分支已经挂载")

    student_encoder = detector.backbone.vision_backbone.trunk.model
    tinyvit = student_encoder.backbone.model
    stage1 = tinyvit.layers[1]
    if not stage1.blocks:
        raise RuntimeError("TinyViT Stage 1没有可捕获的block")
    stage1_channels = int(stage1.dim)
    stage1_resolution = tuple(int(value) for value in stage1.input_resolution)
    output_channels = int(student_encoder.head[-1].out_channels)

    branch = ThinLineDSConvBranch(
        in_channels=stage1_channels,
        out_channels=output_channels,
        branch_channels=branch_channels,
        kernel_size=kernel_size,
        offset_scale=offset_scale,
    )
    reference = next(student_encoder.parameters())
    branch.to(device=reference.device, dtype=reference.dtype)
    detector.add_module("p6_stage1_thin_line_branch", branch)
    detector._p6_stage1_feature = None

    def capture_stage1(_module, _inputs, output):
        batch, tokens, channels = output.shape
        height, width = stage1_resolution
        if tokens != height * width or channels != stage1_channels:
            raise RuntimeError(
                "TinyViT Stage 1形状不匹配："
                f"实际={tuple(output.shape)}，预期tokens={height * width}, "
                f"channels={stage1_channels}"
            )
        detector._p6_stage1_feature = output.transpose(1, 2).reshape(
            batch, channels, height, width,
        )

    hook_handle = stage1.blocks[-1].register_forward_hook(capture_stage1)
    # 这里保存的是已经包含P5 Stage 2残差的forward。
    p5_forward = student_encoder.forward

    def forward_with_p6(_self, images):
        detector._p6_stage1_feature = None
        p5_output = p5_forward(images)
        stage1_feature = detector._p6_stage1_feature
        if stage1_feature is None:
            raise RuntimeError("没有捕获到TinyViT Stage 1特征")
        residual = detector.p6_stage1_thin_line_branch(
            stage1_feature, tuple(p5_output.shape[-2:]),
        )
        return p5_output + residual.to(dtype=p5_output.dtype)

    student_encoder.forward = MethodType(forward_with_p6, student_encoder)
    detector._p6_stage1_hook_handle = hook_handle
    detector._p6_original_p5_forward = p5_forward
    detector.p6_attachment = P6Attachment(
        stage1_channels=stage1_channels,
        stage1_resolution=stage1_resolution,
        output_channels=output_channels,
    )
    return detector.p6_attachment


def restore_p5_then_attach_p6(
    detector: nn.Module,
    checkpoint_path: Path,
    stage1_branch_channels: int = 64,
    stage1_kernel_size: int = 9,
    stage1_offset_scale: float = 1.0,
) -> tuple[dict, P6Attachment]:
    """恢复P5 checkpoint中的Stage 2分支，再零初始化挂载Stage 1分支。"""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"无效P5 checkpoint：{checkpoint_path}")
    meta = payload.get("meta", {})
    if int(meta.get("p5_stage", -1)) != 2:
        raise ValueError(f"P6初始化权重不是P5 Stage 2 checkpoint：{checkpoint_path}")

    attach_p5_dsconv_branch(
        detector,
        branch_channels=int(meta.get("p5_branch_channels", 128)),
        kernel_size=int(meta.get("p5_dsconv_kernel_size", 9)),
        offset_scale=float(meta.get("p5_offset_scale", 1.0)),
    )
    missing, unexpected = detector.load_state_dict(payload["state_dict"], strict=False)
    missing_p5 = [key for key in missing if key.startswith(P5_BRANCH_PREFIX)]
    unexpected_relevant = [
        key for key in unexpected
        if key.startswith(P5_BRANCH_PREFIX) or key.startswith(P6_BRANCH_PREFIX)
    ]
    if missing_p5 or unexpected_relevant:
        raise RuntimeError(
            f"P5初始化权重不匹配：missing={missing_p5}, "
            f"unexpected={unexpected_relevant}"
        )
    attachment = attach_p6_stage1_branch(
        detector,
        branch_channels=stage1_branch_channels,
        kernel_size=stage1_kernel_size,
        offset_scale=stage1_offset_scale,
    )
    return meta, attachment


def attach_multiscale_from_checkpoint(detector: nn.Module, checkpoint_path: Path) -> dict:
    """为评测模型挂载P5/P6两级分支并加载完整权重。"""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    attach_p5_dsconv_branch(
        detector,
        branch_channels=int(meta.get("p5_branch_channels", 128)),
        kernel_size=int(meta.get("p5_dsconv_kernel_size", 9)),
        offset_scale=float(meta.get("p5_offset_scale", 1.0)),
    )
    attach_p6_stage1_branch(
        detector,
        branch_channels=int(meta.get("p6_stage1_branch_channels", 64)),
        kernel_size=int(meta.get("p6_stage1_kernel_size", 9)),
        offset_scale=float(meta.get("p6_stage1_offset_scale", 1.0)),
    )
    return meta
