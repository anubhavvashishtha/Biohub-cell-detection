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
        w_bg: float = 1.0,
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
        cell_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred, gt = self._align_pred_gt(pred, gt)
        
        # Linajea's logic: weight pixels based on GT cutoff (0.01)
        # gt is Gaussian blob, so <0.01 is background
        is_bg = gt < 0.01
        is_fg = ~is_bg
        
        weight = torch.where(is_bg, self.w_bg, self.w_fg)
        
        diff2 = (pred - gt) ** 2
        
        total_pixels = float(diff2.numel())
        
        loss_ci = (weight * diff2).mean()
        
        # Additive components of loss_ci
        loss_bg = (self.w_bg * diff2[is_bg]).sum() / total_pixels if is_bg.any() else torch.tensor(0.0, device=pred.device)
        loss_fg = (self.w_fg * diff2[is_fg]).sum() / total_pixels if is_fg.any() else torch.tensor(0.0, device=pred.device)
        
        return loss_ci, loss_bg, loss_fg

    def _calc_mv_loss(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, pred_indicator: torch.Tensor | None = None) -> torch.Tensor:
        if mask.sum() == 0:
            return pred.sum() * 0.0

        if pred_indicator is not None:
            confidence = pred_indicator.squeeze(1).detach()
        else:
            confidence = torch.ones_like(mask, dtype=pred.dtype)

        weight = (confidence * mask.float()).unsqueeze(1)
        diff2 = (pred - gt) ** 2
        return (weight * diff2).sum() / mask.sum().clamp(min=1)

    def movement_vector_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        cell_mask: torch.Tensor,
        maxima: torch.Tensor | None = None,
        pred_indicator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        pred, gt = self._align_pred_gt(pred, gt)
        cell_mask, _ = self._align_pred_gt(cell_mask, gt[:, :1])
        cell_mask = cell_mask.squeeze(1)

        # LA: Full cell mask
        mask_la = cell_mask > 0
        
        if pred_indicator is not None:
            pred_indicator, _ = self._align_pred_gt(pred_indicator, gt[:, :1])
            
        loss_la = self._calc_mv_loss(pred, gt, mask_la, pred_indicator)

        # LB: Maxima only
        if maxima is not None:
            maxima, _ = self._align_pred_gt(maxima.float(), gt[:, :1])
            mask_lb = maxima.squeeze(1) > 0
        else:
            mask_lb = mask_la
            
        loss_lb = self._calc_mv_loss(pred, gt, mask_lb, pred_indicator)

        mode = self.vector_mask_mode
        if mode == "curriculum":
            # Smooth transition centered at curriculum_epoch.
            # alpha drops from 1.0 (full mask) to 0.0 (maxima only).
            diff = float(self.current_epoch - self.curriculum_epoch)
            # scale=0.5 provides a smooth transition window of ~15 epochs
            alpha = 1.0 / (1.0 + torch.exp(torch.tensor(0.5 * diff))).item()
            loss_mv = alpha * loss_la + (1.0 - alpha) * loss_lb
        elif mode == "maxima":
            alpha = 0.0
            loss_mv = loss_lb
        else:
            alpha = 1.0
            loss_mv = loss_la

        return loss_mv, loss_la, loss_lb, alpha

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

        loss_ci, loss_bg, loss_fg = self.cell_indicator_loss(pred_ci, gt_ci, cell_mask)
        loss_mv, loss_la, loss_lb, alpha = self.movement_vector_loss(
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
            "loss_bg": loss_bg.detach(),
            "loss_fg": loss_fg.detach(),
            "loss_mv": loss_mv.detach(),
            "loss_la": loss_la.detach(),
            "loss_lb": loss_lb.detach(),
            "alpha": torch.tensor(alpha, device=total.device),
        }
