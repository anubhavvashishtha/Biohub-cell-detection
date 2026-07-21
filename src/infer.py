"""Inference pipeline and Kaggle submission export.

Key fix vs. original: the U-Net uses valid padding, so the output spatial
shape is smaller than the input.  All detected (z, y, x) peaks are in the
*network-output* coordinate frame and must be shifted by the center-crop
offset to recover full-volume coordinates before writing to the CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import Config
from src.data.zarr_loader import ZarrVolume, list_zarr_samples, normalize_volume
from src.models.residual_unet3d import ResidualCellTrackingUNet as CellTrackingUNet
from src.tracking.linker import (
    DetectedNode,
    assign_global_node_ids,
    build_tracking_graph,
    extract_peaks_from_frame,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: Config, device: torch.device) -> CellTrackingUNet:
    model = CellTrackingUNet.from_config(config)
    ckpt_path = Path(config.inference.checkpoint)
    if ckpt_path.exists():
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: checkpoint not found at {ckpt_path}, using random weights")
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Crop-offset computation
# ---------------------------------------------------------------------------

def compute_crop_offset(
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Return the (dz, dy, dx) center-crop offset that maps network output
    coordinates back to full-volume (input) coordinates.

    Because the U-Net uses valid (no) padding, the output is centered inside
    the input:
        full_coord = output_coord + offset
    """
    return tuple((i - o) // 2 for i, o in zip(input_shape, output_shape))


# ---------------------------------------------------------------------------
# Per-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_sample(
    model: CellTrackingUNet,
    volume: ZarrVolume,
    device: torch.device,
    normalize_mode: str = "per_volume",
    threshold: float = 0.5,
) -> dict[int, list[DetectedNode]]:
    """Run detection on every timepoint of *volume*.

    Returns ``{timepoint: [DetectedNode, …]}`` with coordinates expressed
    in the *full-volume* voxel space.
    """
    input_shape: tuple[int, int, int] = tuple(volume.shape[1:])   # (Z, Y, X)
    output_shape: tuple[int, int, int] = model.output_shape        # valid-padded output
    crop_offset = compute_crop_offset(input_shape, output_shape)

    frame_detections: dict[int, list[DetectedNode]] = {}

    for t in tqdm(range(volume.num_timepoints), desc=volume.sample_id, leave=False):
        frames = []
        for i in range(-3, 4):
            t_offset = max(0, min(t + i, volume.num_timepoints - 1))
            f = normalize_volume(volume.get_frame(t_offset), mode=normalize_mode)
            frames.append(f)
            
        import numpy as np
        frame_window = np.stack(frames, axis=0) # (7, Z, Y, X)
        x = torch.from_numpy(frame_window).unsqueeze(0).unsqueeze(0).float().to(device)

        outputs = model(x, inference=True)

        ci     = outputs["cell_indicator"][0, 0].cpu().numpy()
        maxima = outputs["maxima"][0].cpu().numpy()
        mv     = outputs["movement_vectors"][0].cpu().numpy()

        nodes = extract_peaks_from_frame(
            ci,
            maxima,
            mv,
            t=t,
            threshold=threshold,
            crop_offset=crop_offset,   # ← maps peaks back to full-volume coords
        )
        frame_detections[t] = nodes

    return frame_detections


# ---------------------------------------------------------------------------
# Submission row builder
# ---------------------------------------------------------------------------

def tracking_result_to_submission_rows(
    sample_id: str,
    frame_detections: dict[int, list[DetectedNode]],
    max_distance_um: float,
    scale: tuple[float, float, float],
    allow_divisions: bool = True,
    start_id: int = 0,
) -> tuple[list[dict], int]:
    """Convert detections to submission CSV rows."""
    frame_detections = assign_global_node_ids(frame_detections)
    result = build_tracking_graph(
        frame_detections,
        max_distance_um=max_distance_um,
        scale=scale,
        allow_divisions=allow_divisions,
    )

    rows: list[dict] = []
    row_id = start_id

    for node in result.nodes:
        rows.append(
            {
                "id":        row_id,
                "dataset":   sample_id,
                "row_type":  "node",
                "node_id":   node.node_id,
                "t":         node.t,
                "z":         node.z,
                "y":         node.y,
                "x":         node.x,
                "source_id": -1,
                "target_id": -1,
            }
        )
        row_id += 1

    for src, tgt in result.edges:
        rows.append(
            {
                "id":        row_id,
                "dataset":   sample_id,
                "row_type":  "edge",
                "node_id":   -1,
                "t":         -1,
                "z":         -1,
                "y":         -1,
                "x":         -1,
                "source_id": src,
                "target_id": tgt,
            }
        )
        row_id += 1

    return rows, row_id


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_inference(config: Config, data_root: str | None = None) -> pd.DataFrame:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(config, device)

    root    = data_root or config.data.data_root
    samples = list_zarr_samples(root)
    if not samples:
        raise FileNotFoundError(f"No .zarr samples under {root}")

    scale: tuple[float, float, float] = tuple(config.data.scale)
    all_rows: list[dict] = []
    row_id = 0

    for sample_path in tqdm(samples, desc="Samples"):
        volume = ZarrVolume(sample_path)
        frame_detections = infer_sample(
            model,
            volume,
            device,
            normalize_mode=config.data.normalize_mode,
            threshold=config.model.nms_threshold,
        )
        rows, row_id = tracking_result_to_submission_rows(
            sample_id=volume.sample_id,
            frame_detections=frame_detections,
            max_distance_um=config.inference.max_link_distance_um,
            scale=scale,
            allow_divisions=True,
            start_id=row_id,
        )
        all_rows.extend(rows)

    columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
               "source_id", "target_id"]
    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df[columns]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and export submission.csv")
    parser.add_argument("--config",    type=str, default="configs/default.yaml")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root (e.g. test/)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output = args.output or config.inference.output
    df     = run_inference(config, data_root=args.data_root)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
