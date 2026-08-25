"""P8输入侧504分辨率细线分支及P7 checkpoint迁移工具。"""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsconv_branch import DynamicSnakeConv2d, P5_BRANCH_PREFIX
from highres_fpn import P7_BRANCH_PREFIX, attach_p7_from_checkpoint
from multiscale_dsconv import P6_BRANCH_PREFIX


P8_BRANCH_PREFIX = "p8_input_line_branch."


class StripConv2d(nn.Module):
    """普通长条卷积对照；接口与单方向DSConv一致。"""

    def __init__(self, in_channels: int, out_channels: int, morphology: str) -> None:
        super().__init__()
        if morphology == "horizontal":
            kernel_size, padding = (1, 9), (0, 4)
        elif morphology == "vertical":
            kernel_size, padding = (9, 1), (4, 0)
        else:
            raise ValueError(f"未知长条卷积方向：{morphology}")
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(DynamicSnakeConv2d._group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InputLineBranch(nn.Module):
    """先在504分辨率提取细线，再下采样并用Stage2语义筛选。"""

    def __init__(
        self,
        operator: str = "dsconv",
        stem_channels: int = 16,
        line_channels: int = 16,
        kernel_size: int = 9,
        offset_scale: float = 1.0,
        output_channels: int = 256,
    ) -> None:
        super().__init__()
        if operator not in {"dsconv", "strip_conv"}:
            raise ValueError(f"未知细线算子：{operator}")
        self.operator = operator
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.stem = nn.Sequential(
            nn.Conv2d(12, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(DynamicSnakeConv2d._group_count(stem_channels), stem_channels),
            nn.GELU(),
        )
        if operator == "dsconv":
            kwargs = dict(
                in_channels=stem_channels,
                out_channels=line_channels,
                kernel_size=kernel_size,
                offset_scale=offset_scale,
            )
            self.horizontal = DynamicSnakeConv2d(**kwargs, morphology="horizontal")
            self.vertical = DynamicSnakeConv2d(**kwargs, morphology="vertical")
        else:
            self.horizontal = StripConv2d(stem_channels, line_channels, "horizontal")
            self.vertical = StripConv2d(stem_channels, line_channels, "vertical")
        detail_channels = line_channels * 2
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(
                detail_channels, detail_channels, kernel_size=3, padding=1,
                groups=detail_channels, bias=False,
            ),
            nn.Conv2d(detail_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, output_channels),
            nn.GELU(),
        )
        self.semantic_gate = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.semantic_gate.weight)
        nn.init.zeros_(self.semantic_gate.bias)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.last_shapes: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _image_tensor(images) -> torch.Tensor:
        if torch.is_tensor(images):
            return images
        if hasattr(images, "tensors"):
            return images.tensors
        if isinstance(images, (list, tuple)) and all(torch.is_tensor(x) for x in images):
            return torch.stack(list(images))
        raise TypeError(f"P8不支持的图像输入类型：{type(images)!r}")

    def forward(
        self, images, semantic_feature: torch.Tensor, output_size: tuple[int, int],
    ) -> torch.Tensor:
        image_tensor = self._image_tensor(images)
        if image_tensor.shape[-2] % 2 or image_tensor.shape[-1] % 2:
            raise RuntimeError(f"P8输入尺寸必须能被2整除：{tuple(image_tensor.shape)}")
        unshuffled = self.pixel_unshuffle(image_tensor)
        stem = self.stem(unshuffled)
        horizontal = self.horizontal(stem)
        vertical = self.vertical(stem)
        detail = torch.cat((horizontal, vertical), dim=1)
        detail = F.interpolate(
            detail, size=output_size, mode="bilinear", align_corners=False,
            antialias=True,
        )
        detail = self.spatial_refine(detail)
        semantic = F.interpolate(
            semantic_feature, size=output_size, mode="bilinear", align_corners=False,
        )
        semantic = torch.sigmoid(self.semantic_gate(semantic.float())).to(detail.dtype)
        output = self.residual_gate * detail * semantic
        self.last_shapes = {
            "input": tuple(image_tensor.shape),
            "unshuffled": tuple(unshuffled.shape),
            "stem": tuple(stem.shape),
            "direction": tuple(horizontal.shape),
            "output": tuple(output.shape),
        }
        return output


def attach_p8_input_line_branch(
    detector: nn.Module,
    operator: str = "dsconv",
    stem_channels: int = 16,
    line_channels: int = 16,
    kernel_size: int = 9,
    offset_scale: float = 1.0,
) -> None:
    if not hasattr(detector, "p7_highres_fpn_adapters"):
        raise RuntimeError("P8要求先挂载P7完整结构")
    if hasattr(detector, "p8_input_line_branch"):
        raise RuntimeError("P8输入侧细线分支已经挂载")
    branch = InputLineBranch(
        operator=operator,
        stem_channels=stem_channels,
        line_channels=line_channels,
        kernel_size=kernel_size,
        offset_scale=offset_scale,
    )
    reference = next(detector.parameters())
    branch.to(device=reference.device, dtype=reference.dtype)
    detector.add_module("p8_input_line_branch", branch)

    neck = detector.backbone.vision_backbone
    p7_forward = neck.forward

    def forward_with_p8(_self, tensor_list):
        sam3_out, sam3_pos, sam2_out, sam2_pos = p7_forward(tensor_list)
        if len(sam3_out) < 3:
            raise RuntimeError(f"P8要求至少3级FPN，实际={len(sam3_out)}")
        residual = detector.p8_input_line_branch(
            tensor_list,
            semantic_feature=sam3_out[1],
            output_size=tuple(sam3_out[0].shape[-2:]),
        )
        sam3_out[0] = sam3_out[0] + residual.to(dtype=sam3_out[0].dtype)
        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = MethodType(forward_with_p8, neck)
    detector._p8_original_p7_forward = p7_forward


def restore_p7_then_attach_p8(
    detector: nn.Module,
    checkpoint_path: Path,
    operator: str = "dsconv",
    stem_channels: int = 16,
    line_channels: int = 16,
    kernel_size: int = 9,
    offset_scale: float = 1.0,
) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"无效P7 checkpoint：{checkpoint_path}")
    meta = payload.get("meta", {})
    if not bool(meta.get("p7_highres_fpn", False)):
        raise ValueError(f"P8初始化权重不是P7 checkpoint：{checkpoint_path}")
    attach_p7_from_checkpoint(detector, checkpoint_path)
    missing, unexpected = detector.load_state_dict(payload["state_dict"], strict=False)
    old_prefixes = (P5_BRANCH_PREFIX, P6_BRANCH_PREFIX, P7_BRANCH_PREFIX)
    missing_old = [key for key in missing if key.startswith(old_prefixes)]
    unexpected_relevant = [
        key for key in unexpected if key.startswith(old_prefixes + (P8_BRANCH_PREFIX,))
    ]
    if missing_old or unexpected_relevant:
        raise RuntimeError(
            f"P7初始化权重不匹配：missing={missing_old}, unexpected={unexpected_relevant}"
        )
    attach_p8_input_line_branch(
        detector, operator, stem_channels, line_channels, kernel_size, offset_scale,
    )
    return meta


def attach_p8_from_checkpoint(detector: nn.Module, checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    attach_p7_from_checkpoint(detector, checkpoint_path)
    attach_p8_input_line_branch(
        detector,
        operator=str(meta.get("p8_operator", "dsconv")),
        stem_channels=int(meta.get("p8_stem_channels", 16)),
        line_channels=int(meta.get("p8_line_channels", 16)),
        kernel_size=int(meta.get("p8_kernel_size", 9)),
        offset_scale=float(meta.get("p8_offset_scale", 1.0)),
    )
    return meta
