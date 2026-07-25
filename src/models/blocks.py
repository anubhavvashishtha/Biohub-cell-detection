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
        shape = tuple(conv_out_size(s, k, padding=0) for s, k in zip(shape, kernel))
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


class Conv4d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple[int, int, int, int]):
        super().__init__()
        self.kt = kernel_size[0]
        pad3d = tuple(k // 2 for k in kernel_size[1:])
        self.convs = nn.ModuleList([
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size[1:], padding=0)
            for _ in range(self.kt)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (N, C, T, Z, Y, X)
        out = []
        for t in range(x.size(2) - self.kt + 1):
            t_out = sum(self.convs[i](x[:, :, t+i]) for i in range(self.kt))
            out.append(t_out)
        return torch.stack(out, dim=2)


class ConvBlock(nn.Module):
    """Single valid Conv + GroupNorm + ReLU. Supports 4D (T,Z,Y,X) or 3D (Z,Y,X) kernels."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple[int, ...]):
        super().__init__()
        self.kernel_size = kernel_size
        groups = min(8, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
            
        if len(kernel_size) == 4:
            self.conv = Conv4d(in_ch, out_ch, kernel_size)
        else:
            pad = tuple(k // 2 for k in kernel_size)
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=0)
            
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.kernel_size) == 3 and x.dim() == 6:
            # Transition from 4D to 3D when T dimension has collapsed to 1
            assert x.size(2) == 1, f"T dimension must be 1 to apply 3D conv, got {x.size(2)}"
            x = x.squeeze(2)
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """Two valid convolutions. Allows specifying two separate kernels."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size1: tuple[int, ...] = (3, 3, 3),
        kernel_size2: tuple[int, ...] = (3, 3, 3),
    ):
        super().__init__()
        mid = out_ch
        self.block = nn.Sequential(
            ConvBlock(in_ch, mid, kernel_size1),
            ConvBlock(mid, out_ch, kernel_size2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    """MaxPool with anisotropic factors. Supports 5D or 6D (applying over spatial only)."""

    def __init__(self, factors: tuple[int, int, int]):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=factors, stride=factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            N, C, T, Z, Y, X = x.shape
            x_reshaped = x.view(N * T, C, Z, Y, X)
            out = self.pool(x_reshaped)
            return out.view(N, C, T, out.size(2), out.size(3), out.size(4))
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
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    """Center-crop spatial dims of 4D, 5D or 6D tensor."""
    if tensor.dim() == 4:
        th, tw = target_shape[-2:]
        _, _, h, w = tensor.shape
        dy = (h - th) // 2
        dx = (w - tw) // 2
        return tensor[:, :, dy : dy + th, dx : dx + tw]
    elif tensor.dim() == 5:
        td, th, tw = target_shape[-3:]
        _, _, d, h, w = tensor.shape
        dz = (d - td) // 2
        dy = (h - th) // 2
        dx = (w - tw) // 2
        return tensor[:, :, dz : dz + td, dy : dy + th, dx : dx + tw]
    elif tensor.dim() == 6:
        td, th, tw = target_shape[-3:]
        _, _, _, d, h, w = tensor.shape
        dz = (d - td) // 2
        dy = (h - th) // 2
        dx = (w - tw) // 2
        return tensor[:, :, :, dz : dz + td, dy : dy + th, dx : dx + tw]
    else:
        raise ValueError(f"Unsupported tensor dimension for center_crop_spatial: {tensor.dim()}")


def crop_tensor_to_shape(
    tensor: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Alias for center_crop_spatial."""
    return center_crop_spatial(tensor, target_shape)
