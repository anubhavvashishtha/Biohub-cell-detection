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
    Supports 4D or 3D kernels via the blocks.ConvBlock equivalent logic.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, ...],
    ) -> None:
        super().__init__()
        from src.models.blocks import Conv4d
        
        self.kernel_size = kernel_size
        groups = min(8, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
            
        if len(kernel_size) == 4:
            self.conv = Conv4d(in_ch, out_ch, kernel_size)
        else:
            pad = tuple(k // 2 for k in kernel_size)
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
            
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.kernel_size) == 3 and x.dim() == 6:
            assert x.size(2) == 1, "T dimension must be 1 to apply 3D conv"
            x = x.squeeze(2)
        return self.act(self.norm(self.conv(x)))


class ResDoubleConv(nn.Module):
    """
    Two valid convolutions with a residual (identity) shortcut.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size1: tuple[int, ...] = (3, 3, 3),
        kernel_size2: tuple[int, ...] = (3, 3, 3),
    ) -> None:
        super().__init__()
        self.conv1 = ResConvBlock(in_ch, out_ch, kernel_size1)
        self.conv2 = ResConvBlock(out_ch, out_ch, kernel_size2)
        
        # 1x1x1 conv if 3D, or 1x1x1x1 conv if 4D shortcut needed.
        # But wait, shortcut is applied after convs. The output of convs is valid padded, 
        # so it's smaller in T (if 4D) and spatial dims.
        # So we can just use 1x1x1 for spatial projection, but we must handle 6D input.
        # Actually, it's easier to just use a regular Conv3d and apply it to spatial dims, 
        # or a 1x1...1 depending on input dim.
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if residual.dim() == 6:
            # Apply 1x1x1 projection over spatial dims only
            N, C, T, Z, Y, X = residual.shape
            residual = residual.view(N * T, C, Z, Y, X)
            residual = self.proj(residual)
            residual = residual.view(N, residual.shape[1], T, Z, Y, X)
        else:
            residual = self.proj(residual)
            
        out = self.conv1(x)
        out = self.conv2(out)
        
        if residual.dim() == 6:
            if out.dim() == 6 and residual.size(2) > out.size(2):
                # Both 6D but out has smaller T
                dt = (residual.size(2) - out.size(2)) // 2
                if dt > 0:
                    residual = residual[:, :, dt:-dt]
            elif out.dim() == 5:
                # Output collapsed to 5D
                # Center crop T to 1
                dt = (residual.size(2) - 1) // 2
                if dt > 0:
                    residual = residual[:, :, dt:-dt] if residual.size(2) > 1 else residual
                # Now squeeze T
                residual = residual.squeeze(2)
            
        # Center-crop residual to match valid-padded output spatial dims
        residual = center_crop_spatial(residual, out.shape[-3:])
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
        
        kernel_size_down = [
            [(3, 3, 3, 3), (3, 3, 3, 3)],
            [(3, 3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)]
        ]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(features):
            kernels = kernel_size_down[i] if i < len(kernel_size_down) else [(3,3,3), (3,3,3)]
            self.encoders.append(ResDoubleConv(in_ch, out_ch, kernel_size1=kernels[0], kernel_size2=kernels[1]))
            self.pools.append(Downsample(self.downsample_factors[i]))
            in_ch = out_ch

        # Bottleneck
        kernels = kernel_size_down[self.levels] if self.levels < len(kernel_size_down) else [(3,3,3), (3,3,3)]
        self.bottleneck = ResDoubleConv(
            features[-1], features[-1] * 2, kernel_size1=kernels[0], kernel_size2=kernels[1]
        )

        # Decoder
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = features[-1] * 2
        for i in reversed(range(self.levels)):
            out_ch = features[i]
            self.upconvs.append(UpsampleBlock(self.downsample_factors[i]))
            self.decoders.append(
                ResDoubleConv(dec_in + out_ch, out_ch, kernel_size1=(3, 3, 3), kernel_size2=(3, 3, 3))
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
            
            if skip.dim() == 6:
                # Center crop T to 1
                dt = (skip.size(2) - 1) // 2
                if dt > 0:
                    skip = skip[:, :, dt:-dt]
                skip = skip.squeeze(2)

            target = tuple(
                min(a, b)
                for a, b in zip(out.shape[-3:], skip.shape[-3:])
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
        feature_multiplier: float = 3.0,
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
            base_kernel=(3, 3, 3),
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
            feature_multiplier=getattr(config.model, 'feature_multiplier', 3.0),
            nms_kernel=config.model.nms_kernel,
            nms_threshold=config.model.nms_threshold,
            input_shape=config.data.input_shape,
        )
