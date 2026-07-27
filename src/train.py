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
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
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
from src.models.residual_unet3d import ResidualCellTrackingUNet


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

def build_dataloaders(config: Config, model: ResidualCellTrackingUNet):
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
    model: ResidualCellTrackingUNet,
    loss_fn: DetectionLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {
        "loss": 0.0, "loss_ci": 0.0, "loss_bg": 0.0, "loss_fg": 0.0, "loss_mv": 0.0,
        "loss_la": 0.0, "loss_lb": 0.0, "alpha": 0.0
    }
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
            if k in losses:
                totals[k] += float(losses[k]) * bs
        n += bs
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: Config, resume_path: str | None = None) -> None:
    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- model ---
    # Ensure model input shape matches training patch size to avoid crop size mismatch
    config.data.input_shape = config.data.patch_size
    model = ResidualCellTrackingUNet.from_config(config).to(device)
    print(
        f"Model output shape: {model.output_shape}  "
        f"(input: {config.data.input_shape})"
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- loss ---
    loss_fn = DetectionLoss(
        w_bg=getattr(config.training, "w_bg", 1.0),
        w_fg=getattr(config.training, "w_fg", 1.0),
        lambda_mv=config.training.lambda_mv,
        vector_mask_mode=config.training.vector_mask_mode,
        curriculum_epoch=config.training.vector_curriculum_epoch,
        output_shape=model.output_shape,
    )

    # --- optimizer + schedulers ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

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
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    # --- data ---
    train_loader, val_loader = build_dataloaders(config, model)
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val    = float("inf")
    val_checks_without_improvement = 0
    start_epoch = 0

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("Loaded optimizer state.")
        if "scaler" in ckpt and ckpt["scaler"] is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        val_checks_without_improvement = ckpt.get("val_checks_without_improvement", 0)
        swa_started = ckpt.get("swa_started", start_epoch >= config.training.swa_start_epoch)

        if "cosine_scheduler" in ckpt and ckpt["cosine_scheduler"] is not None:
            cosine_scheduler.load_state_dict(ckpt["cosine_scheduler"])
            for param_group, lr in zip(optimizer.param_groups, cosine_scheduler.get_last_lr()):
                param_group['lr'] = lr
            print("Loaded cosine_scheduler state.")
        else:
            # Fast-forward cosine_scheduler to start_epoch if resuming an older checkpoint
            for _ in range(start_epoch):
                if not swa_started:
                    cosine_scheduler.step()

        if "swa_scheduler" in ckpt and ckpt["swa_scheduler"] is not None:
            swa_scheduler.load_state_dict(ckpt["swa_scheduler"])
            print("Loaded swa_scheduler state.")
        if "swa_model" in ckpt and ckpt["swa_model"] is not None:
            swa_model.load_state_dict(ckpt["swa_model"])

        print(
            f"Resumed at epoch {start_epoch}, step {global_step}. "
            f"Best val: {best_val:.4f}, Patience: {val_checks_without_improvement}/{config.training.patience}, "
            f"Current LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

    # ------------------------------------------------------------------ #
    #  Main training loop                                                  #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, config.training.epochs):
        loss_fn.set_epoch(epoch)
        model.train()

        # Activate SWA
        if epoch >= config.training.swa_start_epoch and not swa_started:
            swa_started = True
            print(f"SWA activated at epoch {epoch + 1}")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.training.epochs}")
        # Training metrics accumulators
        epoch_train_losses = {"loss": 0.0, "loss_ci": 0.0, "loss_bg": 0.0, "loss_fg": 0.0, "loss_mv": 0.0, "loss_la": 0.0, "loss_lb": 0.0, "alpha": 0.0}
        n_batches = 0

        accum_steps = max(1, getattr(config.training, "gradient_accumulation_steps", 1))
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            loss_fn.current_step = global_step
            volume  = batch["volume"].to(device)
            targets = {
                "cell_indicator":   batch["cell_indicator"].to(device),
                "cell_mask":        batch["cell_mask"].to(device),
                "movement_vectors": batch["movement_vectors"].to(device),
            }

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(volume, inference=False)
                losses  = loss_fn(outputs, targets)
                raw_loss = losses["loss"]
                scaled_loss = raw_loss / accum_steps

            scaler.scale(scaled_loss).backward()

            n_batches += 1
            is_accum_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))

            if is_accum_step:
                # Gradient clipping — unscale first so clip threshold is meaningful
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            for k in epoch_train_losses:
                if k in losses:
                    epoch_train_losses[k] += float(losses[k].detach().item())

            pbar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                ci  =f"{losses['loss_ci']:.4f}",
                bg  =f"{losses.get('loss_bg', 0.0):.4f}",
                fg  =f"{losses.get('loss_fg', 0.0):.4f}",
                mv  =f"{losses['loss_mv']:.4f}",
                lr  =f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # Step-level validation
            is_last_step = (n_batches == len(train_loader))
            if is_accum_step and (global_step % config.training.val_every == 0 or is_last_step):
                val_m = validate(model, loss_fn, val_loader, device)
                latest_val_m = val_m
                print(
                    f"\n  [step {global_step}] val loss={val_m['loss']:.4f} "
                    f"ci={val_m['loss_ci']:.4f} bg={val_m.get('loss_bg', 0.0):.4f} fg={val_m.get('loss_fg', 0.0):.4f} mv={val_m['loss_mv']:.4f} "
                    f"la={val_m.get('loss_la', 0.0):.4f} lb={val_m.get('loss_lb', 0.0):.4f} alpha={val_m.get('alpha', 0.0):.4f}"
                )
                if val_m["loss"] < best_val:
                    best_val = val_m["loss"]
                    val_checks_without_improvement = 0
                    _save_checkpoint(
                        model, optimizer, scaler, config, epoch, global_step, best_val, val_checks_without_improvement, ckpt_dir / "best.pt",
                        cosine_scheduler=cosine_scheduler, swa_scheduler=swa_scheduler, swa_started=swa_started, swa_model=swa_model
                    )
                    print(f"  ↑ New best val loss: {best_val:.4f}")
                else:
                    val_checks_without_improvement += 1
                    print(f"  - No improvement for {val_checks_without_improvement} val checks")
                    
                if config.training.patience > 0 and val_checks_without_improvement >= config.training.patience:
                    print(f"Early stopping triggered after {val_checks_without_improvement} checks without improvement.")
                    break

        if config.training.patience > 0 and val_checks_without_improvement >= config.training.patience:
            break

        # ---- end of epoch ----
        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_train_losses.items()}
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train: Loss={avg_train['loss']:.4f}, CI={avg_train['loss_ci']:.4f}, BG={avg_train.get('loss_bg', 0.0):.4f}, FG={avg_train.get('loss_fg', 0.0):.4f}, MV={avg_train['loss_mv']:.4f}, LA={avg_train['loss_la']:.4f}, LB={avg_train['loss_lb']:.4f}, Alpha={avg_train['alpha']:.4f}")
        print(f"  Val  : Loss={latest_val_m['loss']:.4f}, CI={latest_val_m['loss_ci']:.4f}, BG={latest_val_m.get('loss_bg', 0.0):.4f}, FG={latest_val_m.get('loss_fg', 0.0):.4f}, MV={latest_val_m['loss_mv']:.4f}, LA={latest_val_m.get('loss_la', 0.0):.4f}, LB={latest_val_m.get('loss_lb', 0.0):.4f}, Alpha={latest_val_m.get('alpha', 0.0):.4f}")

        # Save to CSV
        csv_path = ckpt_dir / "training_log.csv"
        write_header = not csv_path.exists()
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "epoch", "train_loss", "train_loss_ci", "train_loss_bg", "train_loss_fg", 
                    "train_loss_mv", "train_loss_la", "train_loss_lb", "train_alpha", 
                    "val_loss", "val_loss_ci", "val_loss_bg", "val_loss_fg", 
                    "val_loss_mv", "val_loss_la", "val_loss_lb", "val_alpha", "lr"
                ])
            writer.writerow([
                epoch + 1,
                avg_train['loss'], avg_train['loss_ci'], avg_train.get('loss_bg', 0.0), avg_train.get('loss_fg', 0.0),
                avg_train['loss_mv'], avg_train['loss_la'], avg_train['loss_lb'], avg_train['alpha'],
                latest_val_m['loss'], latest_val_m['loss_ci'], latest_val_m.get('loss_bg', 0.0), latest_val_m.get('loss_fg', 0.0),
                latest_val_m['loss_mv'], latest_val_m.get('loss_la', 0.0), latest_val_m.get('loss_lb', 0.0), latest_val_m.get('alpha', 0.0),
                optimizer.param_groups[0]['lr']
            ])
            
        # Update metrics graph
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(df["epoch"], df["train_loss"], label="Train")
            plt.plot(df["epoch"], df["val_loss"], label="Val")
            plt.title("Total Loss")
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(df["epoch"], df["train_ci"], label="Train CI")
            plt.plot(df["epoch"], df["val_ci"], label="Val CI")
            plt.title("Cell Indicator Loss")
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.plot(df["epoch"], df["train_la"], label="Train LA (Full)")
            plt.plot(df["epoch"], df["train_lb"], label="Train LB (Maxima)")
            plt.plot(df["epoch"], df["train_mv"], label="Train MV (Combined)", linestyle='--')
            plt.title("Tracking Losses")
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(df["epoch"], df["train_alpha"], label="Alpha")
            plt.title("Curriculum Alpha")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(ckpt_dir / "metrics.png")
            plt.close()
        except ImportError:
            pass # Skip if pandas not installed
            
        # Always save last.pt for easy resuming with patience intact
        _save_checkpoint(
            model, optimizer, scaler, config, epoch, global_step, best_val, val_checks_without_improvement, ckpt_dir / "last.pt",
            cosine_scheduler=cosine_scheduler, swa_scheduler=swa_scheduler, swa_started=swa_started, swa_model=swa_model
        )

        # LR step
        if swa_started:
            swa_scheduler.step()
        else:
            cosine_scheduler.step()
            
        # SWA parameter averaging (ONCE PER EPOCH)
        if swa_started:
            swa_model.update_parameters(model)

        # Epoch-level checkpoint (every 10 epochs or last)
        if (epoch + 1) % 10 == 0 or epoch == config.training.epochs - 1:
            _save_checkpoint(
                model, optimizer, scaler, config, epoch, global_step, best_val, val_checks_without_improvement,
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                cosine_scheduler=cosine_scheduler, swa_scheduler=swa_scheduler, swa_started=swa_started, swa_model=swa_model
            )

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
    model: ResidualCellTrackingUNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Config,
    epoch: int,
    step: int,
    best_val: float,
    val_checks_without_improvement: int,
    path: Path,
    cosine_scheduler: CosineAnnealingLR | None = None,
    swa_scheduler: SWALR | None = None,
    swa_started: bool = False,
    swa_model: AveragedModel | None = None,
) -> None:
    payload = {
        "model":  model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": _config_to_dict(config),
        "epoch":  epoch,
        "step":   step,
        "best_val": best_val,
        "val_checks_without_improvement": val_checks_without_improvement,
        "swa_started": swa_started,
    }
    if cosine_scheduler is not None:
        payload["cosine_scheduler"] = cosine_scheduler.state_dict()
    if swa_scheduler is not None:
        payload["swa_scheduler"] = swa_scheduler.state_dict()
    if swa_model is not None:
        payload["swa_model"] = swa_model.state_dict()
    torch.save(payload, path)


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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., checkpoints/last.pt)",
    )
    args   = parser.parse_args()
    config = Config.from_yaml(args.config)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
