"""Anisotropic 3D convolution blocks with valid padding."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def conv_out_size(
    size: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def pool_out_size(size: int, kernel: int, stride: int, padding: int = 0) -> int:
    return conv_out_size(size, kernel, stride=stride, padding=padding)


def compute_spatial_shape_after_convs(
    input_shape: tuple[int, int, int],
    kernel: tuple[int, int, int],
    num_convs: int = 2,
) -> tuple[int, int, int]:
    shape = input_shape
    for _ in range(num_convs):
        shape = tuple(conv_out_size(s, k) for s, k in zip(shape, kernel))
    return shape


def compute_output_shape(
    input_shape: tuple[int, int, int],
    downsample_factors: Sequence[Sequence[int]],
    base_kernel: tuple[int, int, int] = (1, 3, 3),
    deep_kernel: tuple[int, int, int] = (3, 3, 3),
    num_levels: int | None = None,
) -> tuple[int, int, int]:
    """
    Dry-run spatial dimensions through encoder (DoubleConv + pool per level)
    and decoder (upsample + concat + DoubleConv per level).
    """
    levels = num_levels or len(downsample_factors)
    shape = input_shape
    skip_shapes: list[tuple[int, int, int]] = []

    for i in range(levels):
        kernel = base_kernel if i == 0 else deep_kernel
        shape = compute_spatial_shape_after_convs(shape, kernel, num_convs=2)
        skip_shapes.append(shape)
        pool = downsample_factors[i]
        shape = tuple(pool_out_size(s, k, stride=k) for s, k in zip(shape, pool))

    # Bottleneck convolutions
    shape = compute_spatial_shape_after_convs(shape, (3, 3, 3), num_convs=2)

    for i in reversed(range(levels)):
        pool = downsample_factors[i]
        shape = tuple(s * p for s, p in zip(shape, pool))
        skip = skip_shapes[i]
        # Crop both to minimum shape (matching PyTorch center crop in decoder block)
        shape = tuple(min(a, b) for a, b in zip(shape, skip))
        kernel = base_kernel if i == 0 else deep_kernel
        shape = compute_spatial_shape_after_convs(shape, kernel, num_convs=2)

    return shape


class ConvBlock(nn.Module):
    """Single valid Conv3d + GroupNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple[int, int, int]):
        super().__init__()
        groups = min(8, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=0)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """Two valid convolutions; anisotropic kernel on level 0."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
    ):
        super().__init__()
        mid = out_ch
        self.block = nn.Sequential(
            ConvBlock(in_ch, mid, kernel_size),
            ConvBlock(mid, out_ch, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    """MaxPool with anisotropic factors."""

    def __init__(self, factors: tuple[int, int, int]):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=factors, stride=factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class UpsampleBlock(nn.Module):
    """Nearest upsample by anisotropic factors."""

    def __init__(self, factors: tuple[int, int, int]):
        super().__init__()
        self.scale = factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


def center_crop_spatial(
    tensor: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Center-crop (D, H, W) spatial dims of (B, C, D, H, W) tensor."""
    _, _, d, h, w = tensor.shape
    td, th, tw = target_shape
    dz = (d - td) // 2
    dy = (h - th) // 2
    dx = (w - tw) // 2
    return tensor[:, :, dz : dz + td, dy : dy + th, dx : dx + tw]


def crop_tensor_to_shape(
    tensor: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Alias for center_crop_spatial."""
    return center_crop_spatial(tensor, target_shape)
