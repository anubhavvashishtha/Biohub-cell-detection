"""Anisotropic 3D U-Net with cell indicator and movement vector heads."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.models.blocks import (
    DoubleConv,
    Downsample,
    UpsampleBlock,
    center_crop_spatial,
    compute_output_shape,
    crop_tensor_to_shape,
)
from src.models.nms import NMS3D


class UNet3DEncoderDecoder(nn.Module):
    """Valid-padded anisotropic 3D U-Net backbone."""

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 16,
        downsample_factors: Sequence[Sequence[int]] = ((1, 2, 2), (2, 2, 2), (2, 2, 2)),
    ):
        super().__init__()
        self.downsample_factors = [tuple(f) for f in downsample_factors]
        self.levels = len(self.downsample_factors)

        features = [base_features * (2**i) for i in range(self.levels)]
        self.features = features

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(features):
            kernel = (1, 3, 3) if i == 0 else (3, 3, 3)
            self.encoders.append(DoubleConv(in_ch, out_ch, kernel_size=kernel))
            self.pools.append(Downsample(self.downsample_factors[i]))
            in_ch = out_ch

        bottleneck_kernel = (3, 3, 3)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, kernel_size=bottleneck_kernel)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = features[-1] * 2
        for i in reversed(range(self.levels)):
            out_ch = features[i]
            self.upconvs.append(UpsampleBlock(self.downsample_factors[i]))
            self.decoders.append(
                DoubleConv(dec_in + out_ch, out_ch, kernel_size=(1, 3, 3) if i == 0 else (3, 3, 3))
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
            # Both skip and upsampled out may have different spatial sizes
            # (valid padding can make upsampled size < skip size).
            # Crop both to their element-wise minimum.
            target = tuple(
                min(a, b)
                for a, b in zip(out.shape[2:], skip.shape[2:])
            )
            out  = center_crop_spatial(out,  target)
            skip = center_crop_spatial(skip, target)
            out  = torch.cat([skip, out], dim=1)
            out  = dec(out)

        return out


class CellTrackingUNet(nn.Module):
    """
    3D U-Net for cell detection and movement vector prediction.

    Outputs:
        - cell_indicator: sigmoid heatmap (B, 1, D, H, W)
        - movement_vectors: linear 3-channel displacement (B, 3, D, H, W)
        - maxima: boolean NMS mask
        - raw_cropped: input cropped to output size (training only)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 16,
        downsample_factors: Sequence[Sequence[int]] = ((1, 2, 2), (2, 2, 2), (2, 2, 2)),
        nms_kernel: int = 3,
        nms_threshold: float = 0.5,
        input_shape: Sequence[int] = (64, 256, 256),
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.downsample_factors = [tuple(f) for f in downsample_factors]

        self.backbone = UNet3DEncoderDecoder(
            in_channels=in_channels,
            base_features=base_features,
            downsample_factors=self.downsample_factors,
        )
        feat = self.backbone.out_channels
        self.cell_head = nn.Conv3d(feat, 1, kernel_size=1)
        self.vector_head = nn.Conv3d(feat, 3, kernel_size=1)
        self.nms = NMS3D(kernel_size=nms_kernel, threshold=nms_threshold)
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
        features = self.backbone(x)
        cell_logits = self.cell_head(features)
        cell_indicator = torch.sigmoid(cell_logits)
        movement_vectors = self.vector_head(features)

        if inference:
            cropped_ci = cell_indicator
            maxima = self.nms(cropped_ci)
            return {
                "cell_indicator": cropped_ci,
                "maxima": maxima,
                "movement_vectors": movement_vectors,
            }

        maxima = self.nms(cell_indicator)
        raw_cropped = self.crop_input(x)
        return {
            "cell_indicator": cell_indicator,
            "maxima": maxima,
            "raw_cropped": raw_cropped,
            "movement_vectors": movement_vectors,
        }

    @classmethod
    def from_config(cls, config) -> CellTrackingUNet:
        return cls(
            in_channels=config.model.in_channels,
            base_features=config.model.base_features,
            downsample_factors=config.model.downsample_factors,
            nms_kernel=config.model.nms_kernel,
            nms_threshold=config.model.nms_threshold,
            input_shape=config.data.input_shape,
        )
