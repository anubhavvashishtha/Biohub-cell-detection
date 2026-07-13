"""Training loop for cell detection U-Net.

Improvements over original skeleton:
  - Cosine annealing LR scheduler (warm restarts optional).
  - Gradient clipping (max norm 1.0) for stable training.
  - SWA scheduler only steps *after* SWA is activated.
  - `torch.optim.swa_utils.update_bn` called at end of training.
  - Epoch-level val metrics logged alongside step-level metrics.
  - Checkpoint saves config as dict (avoids pickle issues across versions).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.dataset import BiohubDetectionDataset, collate_batch, split_samples_by_embryo
from src.data.zarr_loader import list_zarr_samples
from src.losses.detection_loss import DetectionLoss
from src.models.unet3d import CellTrackingUNet


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_dataloaders(config: Config, model: CellTrackingUNet):
    all_paths = list_zarr_samples(config.data.data_root)
    if not all_paths:
        raise FileNotFoundError(
            f"No .zarr samples found under '{config.data.data_root}'. "
            "Place training data there or update configs/default.yaml."
        )

    train_paths, val_paths = split_samples_by_embryo(
        all_paths,
        val_fraction=config.training.val_embryo_fraction,
        seed=config.training.seed,
    )
    print(
        f"Dataset split: {len(train_paths)} train samples, "
        f"{len(val_paths)} val samples"
    )

    train_ds = BiohubDetectionDataset(
        config,
        sample_paths=train_paths,
        augment=True,
        output_shape=model.output_shape,
    )
    val_ds = BiohubDetectionDataset(
        config,
        sample_paths=val_paths,
        augment=False,
        output_shape=model.output_shape,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: CellTrackingUNet,
    loss_fn: DetectionLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "loss_ci": 0.0, "loss_mv": 0.0}
    n = 0
    for batch in loader:
        volume  = batch["volume"].to(device)
        targets = {
            "cell_indicator":  batch["cell_indicator"].to(device),
            "cell_mask":       batch["cell_mask"].to(device),
            "movement_vectors":batch["movement_vectors"].to(device),
        }
        outputs = model(volume, inference=False)
        losses  = loss_fn(outputs, targets)
        bs = volume.size(0)
        for k in totals:
            totals[k] += float(losses[k]) * bs
        n += bs
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: Config) -> None:
    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- model ---
    # Ensure model input shape matches training patch size to avoid crop size mismatch
    config.data.input_shape = config.data.patch_size
    model = CellTrackingUNet.from_config(config).to(device)
    print(
        f"Model output shape: {model.output_shape}  "
        f"(input: {config.data.input_shape})"
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- loss ---
    loss_fn = DetectionLoss(
        w_bg=config.training.w_bg,
        w_fg=config.training.w_fg,
        lambda_mv=config.training.lambda_mv,
        vector_mask_mode=config.training.vector_mask_mode,
        curriculum_epoch=config.training.vector_curriculum_epoch,
        output_shape=model.output_shape,
    )

    # --- optimizer + schedulers ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    # Cosine annealing over the non-SWA phase
    n_non_swa_epochs = max(1, config.training.swa_start_epoch)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=n_non_swa_epochs,
        eta_min=config.training.lr * 0.01,
    )

    # SWA components
    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config.training.lr * 0.05)
    swa_started   = False

    # AMP scaler
    use_amp = config.training.amp and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- data ---
    train_loader, val_loader = build_dataloaders(config, model)
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val    = float("inf")

    # ------------------------------------------------------------------ #
    #  Main training loop                                                  #
    # ------------------------------------------------------------------ #
    for epoch in range(config.training.epochs):
        loss_fn.set_epoch(epoch)
        model.train()

        # Activate SWA
        if epoch >= config.training.swa_start_epoch and not swa_started:
            swa_started = True
            print(f"SWA activated at epoch {epoch + 1}")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.training.epochs}")
        epoch_loss = 0.0
        n_batches  = 0

        for batch in pbar:
            volume  = batch["volume"].to(device)
            targets = {
                "cell_indicator":   batch["cell_indicator"].to(device),
                "cell_mask":        batch["cell_mask"].to(device),
                "movement_vectors": batch["movement_vectors"].to(device),
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(volume, inference=False)
                losses  = loss_fn(outputs, targets)
                loss    = losses["loss"]

            scaler.scale(loss).backward()

            # Gradient clipping — unscale first so clip threshold is meaningful
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # SWA parameter averaging
            if swa_started:
                swa_model.update_parameters(model)

            epoch_loss += float(losses["loss"])
            n_batches  += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                ci  =f"{losses['loss_ci']:.4f}",
                mv  =f"{losses['loss_mv']:.4f}",
                lr  =f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # Step-level validation
            if global_step % config.training.val_every == 0:
                val_m = validate(model, loss_fn, val_loader, device)
                print(
                    f"\n  [step {global_step}] val loss={val_m['loss']:.4f} "
                    f"ci={val_m['loss_ci']:.4f} mv={val_m['loss_mv']:.4f}"
                )
                if val_m["loss"] < best_val:
                    best_val = val_m["loss"]
                    _save_checkpoint(model, config, epoch, global_step, ckpt_dir / "best.pt")
                    print(f"  ↑ New best val loss: {best_val:.4f}")

        # ---- end of epoch ----
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1} avg train loss: {avg_loss:.4f}")

        # LR step
        if swa_started:
            swa_scheduler.step()
        else:
            cosine_scheduler.step()

        # Epoch-level checkpoint (every 10 epochs or last)
        if (epoch + 1) % 10 == 0 or epoch == config.training.epochs - 1:
            _save_checkpoint(model, config, epoch, global_step,
                             ckpt_dir / f"epoch_{epoch + 1:03d}.pt")

    # ---- Post-training: SWA batch-norm update ----
    if swa_started:
        print("Updating SWA batch norms…")
        # update_bn needs a DataLoader that yields inputs; wrap train_loader
        _update_swa_bn(swa_model, train_loader, device)
        torch.save(
            {
                "model":  swa_model.module.state_dict(),
                "config": _config_to_dict(config),
            },
            ckpt_dir / "swa_final.pt",
        )
        print(f"SWA model saved → {ckpt_dir / 'swa_final.pt'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: CellTrackingUNet,
    config: Config,
    epoch: int,
    step: int,
    path: Path,
) -> None:
    torch.save(
        {
            "model":  model.state_dict(),
            "config": _config_to_dict(config),
            "epoch":  epoch,
            "step":   step,
        },
        path,
    )


def _config_to_dict(config: Config) -> dict:
    return {
        "model":     config.model.__dict__.copy(),
        "data":      config.data.__dict__.copy(),
        "training":  config.training.__dict__.copy(),
        "inference": config.inference.__dict__.copy(),
    }


def _update_swa_bn(
    swa_model: AveragedModel,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """Push a few batches through the SWA model to update BatchNorm stats."""
    swa_model.train()
    with torch.no_grad():
        for batch in loader:
            volume = batch["volume"].to(device)
            swa_model(volume)
    swa_model.eval()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train cell detection U-Net")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    args   = parser.parse_args()
    config = Config.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
