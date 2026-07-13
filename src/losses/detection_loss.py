"""Combined loss for cell indicator and movement vector prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import center_crop_spatial


class DetectionLoss(nn.Module):
    """
    Weighted MSE for cell indicator heatmap and masked movement vectors.

    vector_mask_mode:
        - 'full': use cell_mask
        - 'maxima': use NMS maxima mask
        - 'curriculum': switch from full to maxima after curriculum_epoch
    """

    def __init__(
        self,
        w_bg: float = 10.0,
        w_fg: float = 1.0,
        lambda_mv: float = 1.0,
        vector_mask_mode: str = "full",
        curriculum_epoch: int = 50,
        output_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.w_bg = w_bg
        self.w_fg = w_fg
        self.lambda_mv = lambda_mv
        self.vector_mask_mode = vector_mask_mode
        self.curriculum_epoch = curriculum_epoch
        self.output_shape = output_shape
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def _align_pred_gt(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred.shape[2:] == gt.shape[2:]:
            return pred, gt
        if self.output_shape is not None:
            pred = center_crop_spatial(pred, self.output_shape)
            gt = center_crop_spatial(gt, self.output_shape)
        else:
            # center-crop both to smallest common shape
            target = tuple(min(a, b) for a, b in zip(pred.shape[2:], gt.shape[2:]))
            pred = center_crop_spatial(pred, target)
            gt = center_crop_spatial(gt, target)
        return pred, gt

    def cell_indicator_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        pred, gt = self._align_pred_gt(pred, gt)
        weight = torch.where(gt < 0.5, self.w_bg, self.w_fg)
        return (weight * (pred - gt) ** 2).mean()

    def movement_vector_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        cell_mask: torch.Tensor,
        maxima: torch.Tensor | None = None,
        pred_indicator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred, gt = self._align_pred_gt(pred, gt)
        cell_mask, _ = self._align_pred_gt(cell_mask, gt[:, :1])
        cell_mask = cell_mask.squeeze(1)

        mode = self.vector_mask_mode
        if mode == "curriculum" and self.current_epoch >= self.curriculum_epoch:
            mode = "maxima"

        if mode == "maxima" and maxima is not None:
            maxima, _ = self._align_pred_gt(maxima.float(), gt[:, :1])
            mask = maxima.squeeze(1) > 0
        else:
            mask = cell_mask > 0

        if mask.sum() == 0:
            return pred.sum() * 0.0

        if pred_indicator is not None:
            pred_indicator, _ = self._align_pred_gt(pred_indicator, gt[:, :1])
            confidence = pred_indicator.squeeze(1).detach()
        else:
            confidence = torch.ones_like(mask, dtype=pred.dtype)

        weight = (confidence * mask.float()).unsqueeze(1)
        diff2 = (pred - gt) ** 2
        return (weight * diff2).sum() / mask.sum().clamp(min=1)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pred_ci = outputs["cell_indicator"]
        gt_ci = targets["cell_indicator"]
        pred_mv = outputs["movement_vectors"]
        gt_mv = targets["movement_vectors"]
        cell_mask = targets["cell_mask"]
        maxima = outputs.get("maxima")

        loss_ci = self.cell_indicator_loss(pred_ci, gt_ci)
        loss_mv = self.movement_vector_loss(
            pred_mv,
            gt_mv,
            cell_mask,
            maxima=maxima,
            pred_indicator=pred_ci,
        )
        total = loss_ci + self.lambda_mv * loss_mv
        return {
            "loss": total,
            "loss_ci": loss_ci.detach(),
            "loss_mv": loss_mv.detach(),
        }
