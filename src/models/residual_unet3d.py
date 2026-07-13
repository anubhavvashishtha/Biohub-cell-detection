"""Residual 3D U-Net variant for larger GPU budgets.

Drop-in replacement for CellTrackingUNet with residual connections in every
double-conv block, which improves gradient flow and allows deeper networks.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.models.blocks import (
    Downsample,
    UpsampleBlock,
    center_crop_spatial,
    compute_output_shape,
    crop_tensor_to_shape,
)
from src.models.nms import NMS3D


# ---------------------------------------------------------------------------
# Residual building block
# ---------------------------------------------------------------------------

class ResConvBlock(nn.Module):
    """
    Single convolution with GroupNorm + GELU activation.

    Uses GELU instead of ReLU for slightly smoother gradient landscape.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        groups = min(8, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=0)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResDoubleConv(nn.Module):
    """
    Two valid convolutions with a residual (identity) shortcut.

    The shortcut uses a 1×1×1 conv when channel sizes differ (or always,
    because valid padding changes spatial size and a direct identity
    shortcut would not match).  We center-crop the shortcut to match the
    output spatial dims.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
    ) -> None:
        super().__init__()
        self.conv1 = ResConvBlock(in_ch, out_ch, kernel_size)
        self.conv2 = ResConvBlock(out_ch, out_ch, kernel_size)
        # Project input channels → output channels for the residual path
        self.proj  = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.conv1(x)
        out = self.conv2(out)
        # Center-crop residual to match valid-padded output
        residual = center_crop_spatial(residual, out.shape[2:])
        return out + residual


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class ResUNet3DEncoderDecoder(nn.Module):
    """Valid-padded anisotropic 3D U-Net backbone with residual blocks."""

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 24,
        downsample_factors: Sequence[Sequence[int]] = ((1, 2, 2), (2, 2, 2), (2, 2, 2)),
        feature_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.downsample_factors = [tuple(f) for f in downsample_factors]
        self.levels = len(self.downsample_factors)

        features = [
            int(base_features * (feature_multiplier ** i))
            for i in range(self.levels)
        ]
        self.features = features

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(features):
            kernel = (1, 3, 3) if i == 0 else (3, 3, 3)
            self.encoders.append(ResDoubleConv(in_ch, out_ch, kernel_size=kernel))
            self.pools.append(Downsample(self.downsample_factors[i]))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ResDoubleConv(
            features[-1], features[-1] * 2, kernel_size=(3, 3, 3)
        )

        # Decoder
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = features[-1] * 2
        for i in reversed(range(self.levels)):
            out_ch = features[i]
            kernel = (1, 3, 3) if i == 0 else (3, 3, 3)
            self.upconvs.append(UpsampleBlock(self.downsample_factors[i]))
            self.decoders.append(
                ResDoubleConv(dec_in + out_ch, out_ch, kernel_size=kernel)
            )
            dec_in = out_ch

        self.out_channels = features[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        out = x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        out = self.bottleneck(out)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = up(out)
            target = tuple(
                min(a, b)
                for a, b in zip(out.shape[2:], skip.shape[2:])
            )
            out  = center_crop_spatial(out,  target)
            skip = center_crop_spatial(skip, target)
            out  = torch.cat([skip, out], dim=1)
            out  = dec(out)

        return out


# ---------------------------------------------------------------------------
# Full model (compatible API with CellTrackingUNet)
# ---------------------------------------------------------------------------

class ResidualCellTrackingUNet(nn.Module):
    """
    Residual 3D U-Net for cell detection + movement vector prediction.

    API is identical to CellTrackingUNet so it can be swapped in.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 24,
        downsample_factors: Sequence[Sequence[int]] = ((1, 2, 2), (2, 2, 2), (2, 2, 2)),
        feature_multiplier: float = 2.0,
        nms_kernel: int = 3,
        nms_threshold: float = 0.5,
        input_shape: Sequence[int] = (64, 256, 256),
    ) -> None:
        super().__init__()
        self.input_shape        = tuple(input_shape)
        self.downsample_factors = [tuple(f) for f in downsample_factors]

        self.backbone = ResUNet3DEncoderDecoder(
            in_channels=in_channels,
            base_features=base_features,
            downsample_factors=self.downsample_factors,
            feature_multiplier=feature_multiplier,
        )
        feat = self.backbone.out_channels
        self.cell_head   = nn.Conv3d(feat, 1, kernel_size=1)
        self.vector_head = nn.Conv3d(feat, 3, kernel_size=1)
        self.nms          = NMS3D(kernel_size=nms_kernel, threshold=nms_threshold)
        self.nms_threshold = nms_threshold

        self.output_shape = compute_output_shape(
            self.input_shape,
            self.downsample_factors,
        )

    def crop_input(self, x: torch.Tensor) -> torch.Tensor:
        return crop_tensor_to_shape(x, self.output_shape)

    def forward(
        self,
        x: torch.Tensor,
        inference: bool = False,
    ) -> dict[str, torch.Tensor]:
        features        = self.backbone(x)
        cell_logits     = self.cell_head(features)
        cell_indicator  = torch.sigmoid(cell_logits)
        movement_vectors = self.vector_head(features)

        if inference:
            maxima = self.nms(cell_indicator)
            return {
                "cell_indicator":   cell_indicator,
                "maxima":           maxima,
                "movement_vectors": movement_vectors,
            }

        maxima      = self.nms(cell_indicator)
        raw_cropped = self.crop_input(x)
        return {
            "cell_indicator":   cell_indicator,
            "maxima":           maxima,
            "raw_cropped":      raw_cropped,
            "movement_vectors": movement_vectors,
        }

    @classmethod
    def from_config(cls, config) -> ResidualCellTrackingUNet:
        return cls(
            in_channels=config.model.in_channels,
            base_features=config.model.base_features,
            downsample_factors=config.model.downsample_factors,
            nms_kernel=config.model.nms_kernel,
            nms_threshold=config.model.nms_threshold,
            input_shape=config.data.input_shape,
        )
