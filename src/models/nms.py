"""3D max-pool non-maximum suppression for cell indicator maps."""

from __future__ import annotations

import torch
import torch.nn as nn


class NMS3D(nn.Module):
    """
    Find local maxima in a 3D heatmap via max pooling.

    A voxel is a peak if its value equals the local max-pool output and exceeds threshold.
    """

    def __init__(self, kernel_size: int = 3, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        padding = kernel_size // 2
        self.pool = nn.MaxPool3d(
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: (B, 1, D, H, W) or (B, D, H, W) sigmoid activations

        Returns:
            Boolean mask of local maxima, same spatial shape as input.
        """
        if heatmap.dim() == 4:
            heatmap = heatmap.unsqueeze(1)

        pooled = self.pool(heatmap)
        maxima = (heatmap == pooled) & (heatmap > self.threshold)
        return maxima.squeeze(1) if maxima.shape[1] == 1 else maxima
