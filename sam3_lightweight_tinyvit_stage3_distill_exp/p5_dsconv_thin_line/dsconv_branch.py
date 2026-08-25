"""P5动态蛇形卷积细线分支及TinyViT Stage 2接入工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F


P5_BRANCH_PREFIX = "p5_thin_line_branch."


class DynamicSnakeConv2d(nn.Module):
    """沿单一主轴排列采样点，并在垂直方向累积可学习偏移。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        morphology: str = "horizontal",
        offset_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("DSConv kernel_size必须是大于等于3的奇数")
        if morphology not in {"horizontal", "vertical"}:
            raise ValueError(f"未知DSConv方向：{morphology}")
        self.kernel_size = int(kernel_size)
        self.morphology = morphology
        self.offset_scale = float(offset_scale)
        self.offset = nn.Conv2d(
            in_channels, self.kernel_size, kernel_size=3, padding=1,
        )
        self.projection = nn.Conv2d(
            in_channels * self.kernel_size, out_channels, kernel_size=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(self._group_count(out_channels), out_channels)
        self.activation = nn.GELU()
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    @staticmethod
    def _group_count(channels: int) -> int:
        for groups in (32, 16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def _cumulative_offsets(self, raw: torch.Tensor) -> torch.Tensor:
        offsets = torch.tanh(raw) * self.offset_scale
        center = self.kernel_size // 2
        left = torch.flip(
            torch.cumsum(torch.flip(offsets[:, :center], dims=(1,)), dim=1),
            dims=(1,),
        )
        middle = torch.zeros_like(offsets[:, center:center + 1])
        right = torch.cumsum(offsets[:, center + 1:], dim=1)
        return torch.cat((left, middle, right), dim=1)

    @staticmethod
    def _normalize(coordinates: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coordinates)
        return coordinates.mul(2.0 / float(size - 1)).sub(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        offsets = self._cumulative_offsets(self.offset(x).float())
        y_base, x_base = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=torch.float32),
            torch.arange(width, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        center = self.kernel_size // 2
        displacement = torch.arange(
            -center, center + 1, device=x.device, dtype=torch.float32,
        ).view(1, self.kernel_size, 1, 1)
        x_base = x_base.view(1, 1, height, width)
        y_base = y_base.view(1, 1, height, width)
        if self.morphology == "horizontal":
            x_coordinates = x_base + displacement
            y_coordinates = y_base + offsets
        else:
            x_coordinates = x_base + offsets
            y_coordinates = y_base + displacement
        x_coordinates = x_coordinates.expand(batch, -1, -1, -1)
        y_coordinates = y_coordinates.expand(batch, -1, -1, -1)
        # 把K组采样网格拼到输出宽度，一次grid_sample完成，减少算子启动开销。
        grid = torch.stack(
            (
                self._normalize(x_coordinates, width),
                self._normalize(y_coordinates, height),
            ),
            dim=-1,
        )
        grid = grid.permute(0, 2, 1, 3, 4).reshape(
            batch, height, self.kernel_size * width, 2,
        )
        sampled = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        features = sampled.reshape(
            batch, x.shape[1], height, self.kernel_size, width,
        ).permute(0, 3, 1, 2, 4).reshape(
            batch, self.kernel_size * x.shape[1], height, width,
        )
        return self.activation(self.norm(self.projection(features)))


class ThinLineDSConvBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_channels: int = 128,
        kernel_size: int = 9,
        offset_scale: float = 1.0,
    ) -> None:
        super().__init__()
        kwargs = {
            "in_channels": in_channels,
            "out_channels": branch_channels,
            "kernel_size": kernel_size,
            "offset_scale": offset_scale,
        }
        self.horizontal = DynamicSnakeConv2d(
            **kwargs, morphology="horizontal",
        )
        self.vertical = DynamicSnakeConv2d(
            **kwargs, morphology="vertical",
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(DynamicSnakeConv2d._group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.gate = nn.Parameter(torch.zeros(()))
        self.last_input_shape: tuple[int, ...] | None = None
        self.last_output_shape: tuple[int, ...] | None = None

    def forward(
        self, stage2_feature: torch.Tensor, output_size: tuple[int, int],
    ) -> torch.Tensor:
        self.last_input_shape = tuple(stage2_feature.shape)
        horizontal = self.horizontal(stage2_feature)
        vertical = self.vertical(stage2_feature)
        output = self.fusion(torch.cat((horizontal, vertical), dim=1))
        if output.shape[-2:] != output_size:
            output = F.interpolate(
                output, size=output_size, mode="bilinear", align_corners=False,
            )
        self.last_output_shape = tuple(output.shape)
        return self.gate * output


@dataclass(frozen=True)
class P5Attachment:
    stage2_channels: int
    stage2_resolution: tuple[int, int]
    output_channels: int


def attach_p5_dsconv_branch(
    detector: nn.Module,
    branch_channels: int = 128,
    kernel_size: int = 9,
    offset_scale: float = 1.0,
) -> P5Attachment:
    """捕获TinyViT Stage 2下采样前特征，并融合到学生投影头输出。"""
    if hasattr(detector, "p5_thin_line_branch"):
        raise RuntimeError("P5 DSConv分支已经挂载")
    vision_root = detector.backbone.vision_backbone
    student_encoder = vision_root.trunk.model
    tinyvit = student_encoder.backbone.model
    stage2 = tinyvit.layers[2]
    if not stage2.blocks:
        raise RuntimeError("TinyViT Stage 2没有可捕获的block")
    stage2_channels = int(stage2.dim)
    stage2_resolution = tuple(int(value) for value in stage2.input_resolution)
    output_channels = int(student_encoder.head[-1].out_channels)
    branch = ThinLineDSConvBranch(
        in_channels=stage2_channels,
        out_channels=output_channels,
        branch_channels=branch_channels,
        kernel_size=kernel_size,
        offset_scale=offset_scale,
    )
    reference = next(student_encoder.parameters())
    branch.to(device=reference.device, dtype=reference.dtype)
    detector.add_module("p5_thin_line_branch", branch)
    detector._p5_stage2_feature = None

    def capture_stage2(_module, _inputs, output):
        batch, tokens, channels = output.shape
        height, width = stage2_resolution
        if tokens != height * width or channels != stage2_channels:
            raise RuntimeError(
                "TinyViT Stage 2形状不匹配："
                f"实际={tuple(output.shape)}，预期tokens={height * width}, "
                f"channels={stage2_channels}"
            )
        detector._p5_stage2_feature = output.transpose(1, 2).reshape(
            batch, channels, height, width,
        )

    hook_handle = stage2.blocks[-1].register_forward_hook(capture_stage2)
    original_forward = student_encoder.forward

    def forward_with_p5(_self, images):
        detector._p5_stage2_feature = None
        base = original_forward(images)
        stage2_feature = detector._p5_stage2_feature
        if stage2_feature is None:
            raise RuntimeError("没有捕获到TinyViT Stage 2特征")
        residual = detector.p5_thin_line_branch(
            stage2_feature, tuple(base.shape[-2:]),
        )
        return base + residual.to(dtype=base.dtype)

    student_encoder.forward = MethodType(forward_with_p5, student_encoder)
    detector._p5_stage2_hook_handle = hook_handle
    detector._p5_original_student_forward = original_forward
    detector.p5_attachment = P5Attachment(
        stage2_channels=stage2_channels,
        stage2_resolution=stage2_resolution,
        output_channels=output_channels,
    )
    return detector.p5_attachment


def p5_branch_state_dict(detector: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in detector.state_dict().items()
        if name.startswith(P5_BRANCH_PREFIX)
    }


def load_p5_checkpoint(
    detector: nn.Module, checkpoint_path: Path,
) -> dict:
    """给已经加载P2权重的detector挂载并加载P5分支。"""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"无效P5 checkpoint：{checkpoint_path}")
    meta = payload.get("meta", {})
    attach_p5_dsconv_branch(
        detector,
        branch_channels=int(meta.get("p5_branch_channels", 128)),
        kernel_size=int(meta.get("p5_dsconv_kernel_size", 9)),
        offset_scale=float(meta.get("p5_offset_scale", 1.0)),
    )
    missing, unexpected = detector.load_state_dict(
        payload["state_dict"], strict=False,
    )
    missing_branch = [
        key for key in missing if key.startswith(P5_BRANCH_PREFIX)
    ]
    unexpected_relevant = [
        key for key in unexpected
        if key.startswith(P5_BRANCH_PREFIX)
        or "parametrizations" in key
        or key.startswith(("dot_prod_scoring.", "segmentation_head."))
    ]
    if missing_branch or unexpected_relevant:
        raise RuntimeError(
            "P5 checkpoint不匹配："
            f"missing_branch={missing_branch}, "
            f"unexpected={unexpected_relevant}"
        )
    return meta
