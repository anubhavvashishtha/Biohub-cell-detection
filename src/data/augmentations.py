"""Gunpowder-style 3D augmentations for microscopy volumes and GT."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from src.data.gt_generator import GroundTruth


@dataclass
class AugmentConfig:
    flip_prob: float = 0.5
    transpose_prob: float = 0.25
    shift_max: tuple[int, int, int] = (2, 8, 8)
    intensity_scale_range: tuple[float, float] = (0.8, 1.2)
    intensity_offset_range: tuple[float, float] = (-0.1, 0.1)
    gaussian_noise_std: float = 0.05
    speckle_std: float = 0.05
    salt_pepper_prob: float = 0.001
    elastic_alpha: float = 8.0
    elastic_sigma: float = 3.0
    elastic_prob: float = 0.3
    zoom_z_range: tuple[float, float] = (0.95, 1.05)
    zoom_xy_range: tuple[float, float] = (0.9, 1.1)
    zoom_prob: float = 0.3


# ---------------------------------------------------------------------------
# Mirror / Transpose
# ---------------------------------------------------------------------------

def _random_flip(
    volume: np.ndarray,
    gt: GroundTruth,
    axis: int,
) -> tuple[np.ndarray, GroundTruth]:
    """
    Flip volume (Z, Y, X) and all GT arrays along spatial *axis* ∈ {0, 1, 2}.

    Movement-vector channels are (dz, dy, dx) → channel index matches axis.
    Flipping spatial axis *a* negates the displacement component for that axis.
    """
    assert axis in (0, 1, 2), f"axis must be 0, 1 or 2, got {axis}"

    v_axis = axis + 1 if volume.ndim == 4 else axis
    vol = np.flip(volume, axis=v_axis).copy()
    ci  = np.flip(gt.cell_indicator, axis=axis).copy()
    cc  = np.flip(gt.cell_center,    axis=axis).copy()
    cm  = np.flip(gt.cell_mask,      axis=axis).copy()
    mv  = np.flip(gt.movement_vectors, axis=axis + 1).copy()  # mv is (3, Z, Y, X)
    mv[axis] *= -1  # negate the displacement component for this axis

    return vol, GroundTruth(ci, cc, cm, mv)


def _transpose_yx(
    volume: np.ndarray,
    gt: GroundTruth,
) -> tuple[np.ndarray, GroundTruth]:
    """
    Swap Y ↔ X axes.  Volume: (Z, Y, X) → (Z, X, Y).
    Movement-vector channels: swap dy (channel 1) and dx (channel 2).
    """
    if volume.ndim == 4:
        vol = np.transpose(volume, (0, 1, 3, 2)).copy()
    else:
        vol = np.transpose(volume, (0, 2, 1)).copy()  # (Z, Y, X) → (Z, X, Y)
    ci  = np.transpose(gt.cell_indicator, (0, 2, 1)).copy()
    cc  = np.transpose(gt.cell_center,    (0, 2, 1)).copy()
    cm  = np.transpose(gt.cell_mask,      (0, 2, 1)).copy()
    # mv shape: (3, Z, Y, X) → (3, Z, X, Y)
    mv  = np.transpose(gt.movement_vectors, (0, 1, 3, 2)).copy()
    # swap dy and dx channel values
    mv[1], mv[2] = mv[2].copy(), mv[1].copy()

    return vol, GroundTruth(ci, cc, cm, mv)


# ---------------------------------------------------------------------------
# Random shift
# ---------------------------------------------------------------------------

def _random_shift(
    volume: np.ndarray,
    gt: GroundTruth,
    max_shift: tuple[int, int, int],
) -> tuple[np.ndarray, GroundTruth]:
    dz = random.randint(-max_shift[0], max_shift[0])
    dy = random.randint(-max_shift[1], max_shift[1])
    dx = random.randint(-max_shift[2], max_shift[2])
    return _shift_arrays(volume, gt, dz, dy, dx)


def _shift_arrays(
    volume: np.ndarray,
    gt: GroundTruth,
    dz: int,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, GroundTruth]:
    """Roll volume and GT by (dz, dy, dx) voxels, filling vacated regions with 0."""
    sz, sy, sx = volume.shape[-3:]

    def _shift_spatial(arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr)
        z0s, z1s = max(0, -dz), min(sz, sz - dz)
        y0s, y1s = max(0, -dy), min(sy, sy - dy)
        x0s, x1s = max(0, -dx), min(sx, sx - dx)
        z0d = max(0, dz)
        y0d = max(0, dy)
        x0d = max(0, dx)
        z1d = z0d + (z1s - z0s)
        y1d = y0d + (y1s - y0s)
        x1d = x0d + (x1s - x0s)
        out[z0d:z1d, y0d:y1d, x0d:x1d] = arr[z0s:z1s, y0s:y1s, x0s:x1s]
        return out

    def _shift_4d(arr: np.ndarray) -> np.ndarray:
        return np.stack([_shift_spatial(arr[c]) for c in range(arr.shape[0])], axis=0)

    if volume.ndim == 4:
        vol = _shift_4d(volume)
    else:
        vol = _shift_spatial(volume)
        
    ci  = _shift_spatial(gt.cell_indicator)
    cc  = _shift_spatial(gt.cell_center)
    cm  = _shift_spatial(gt.cell_mask)
    mv  = _shift_4d(gt.movement_vectors)  # displacement magnitude unchanged by shift
    return vol, GroundTruth(ci, cc, cm, mv)


# ---------------------------------------------------------------------------
# Intensity augmentation
# ---------------------------------------------------------------------------

def _intensity_augment(volume: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    vol = volume.astype(np.float32)
    scale  = random.uniform(*cfg.intensity_scale_range)
    offset = random.uniform(*cfg.intensity_offset_range)
    vol = vol * scale + offset

    if cfg.gaussian_noise_std > 0:
        vol += np.random.normal(0, cfg.gaussian_noise_std, vol.shape).astype(np.float32)

    if cfg.speckle_std > 0:
        vol += vol * np.random.normal(0, cfg.speckle_std, vol.shape).astype(np.float32)

    if cfg.salt_pepper_prob > 0:
        mask = np.random.random(vol.shape)
        vol[mask < cfg.salt_pepper_prob / 2]       = 0.0
        vol[mask > 1 - cfg.salt_pepper_prob / 2]   = 1.0

    return np.clip(vol, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Elastic deformation
# ---------------------------------------------------------------------------

def _elastic_deform(
    volume: np.ndarray,
    gt: GroundTruth,
    alpha: float,
    sigma: float,
) -> tuple[np.ndarray, GroundTruth]:
    """Random elastic deformation via dense displacement fields."""
    dz, dy, dx = volume.shape[-3:]

    # Low-resolution random fields, then smooth and upsample
    rz = max(2, dz // 4)
    ry = max(2, dy // 8)
    rx = max(2, dx // 8)

    rz_c = np.linspace(0, dz - 1, rz)
    ry_c = np.linspace(0, dy - 1, ry)
    rx_c = np.linspace(0, dx - 1, rx)

    def _make_field():
        raw = gaussian_filter(
            (np.random.rand(rz, ry, rx) * 2 - 1), sigma, mode="reflect"
        ) * alpha
        # Resample onto full grid
        z_grid = np.linspace(0, dz - 1, dz)
        y_grid = np.linspace(0, dy - 1, dy)
        x_grid = np.linspace(0, dx - 1, dx)
        zg, yg, xg = np.meshgrid(z_grid, y_grid, x_grid, indexing="ij")
        lo_coords = [
            map_coordinates(
                np.linspace(0, rz - 1, rz)[:, None, None] * np.ones((rz, ry, rx)),
                [zg * (rz - 1) / max(dz - 1, 1),
                 yg * (ry - 1) / max(dy - 1, 1),
                 xg * (rx - 1) / max(dx - 1, 1)],
                order=0, mode="nearest",
            )
        ]
        return map_coordinates(raw, [
            zg * (rz - 1) / max(dz - 1, 1),
            yg * (ry - 1) / max(dy - 1, 1),
            xg * (rx - 1) / max(dx - 1, 1),
        ], order=1, mode="nearest")

    # Full-resolution coordinate grids
    z_grid = np.linspace(0, dz - 1, dz)
    y_grid = np.linspace(0, dy - 1, dy)
    x_grid = np.linspace(0, dx - 1, dx)
    zg, yg, xg = np.meshgrid(z_grid, y_grid, x_grid, indexing="ij")

    def _disp_field():
        raw  = (np.random.rand(rz, ry, rx) * 2 - 1)
        raw  = gaussian_filter(raw, sigma, mode="reflect") * alpha
        return map_coordinates(raw, [
            zg * (rz - 1) / max(dz - 1, 1),
            yg * (ry - 1) / max(dy - 1, 1),
            xg * (rx - 1) / max(dx - 1, 1),
        ], order=1, mode="nearest")

    disp_z = _disp_field()
    disp_y = _disp_field()
    disp_x = _disp_field()

    coords = np.array([zg + disp_z, yg + disp_y, xg + disp_x])

    def _warp(arr: np.ndarray, order: int) -> np.ndarray:
        return map_coordinates(arr, coords, order=order, mode="nearest").reshape((dz, dy, dx)).astype(np.float32)

    if volume.ndim == 4:
        vol = np.stack([_warp(volume[t], 1) for t in range(volume.shape[0])], axis=0)
    else:
        vol = _warp(volume, 1)
        
    ci  = _warp(gt.cell_indicator, 1)
    cc  = _warp(gt.cell_center,    0)
    cm  = _warp(gt.cell_mask,      0)
    mv  = np.stack([_warp(gt.movement_vectors[c], 1) for c in range(3)], axis=0)

    return vol, GroundTruth(ci, cc, cm, mv)


# ---------------------------------------------------------------------------
# Anisotropic zoom
# ---------------------------------------------------------------------------

def _anisotropic_zoom(
    volume: np.ndarray,
    gt: GroundTruth,
    z_factor: float,
    xy_factor: float,
) -> tuple[np.ndarray, GroundTruth]:
    shape = volume.shape
    factors = (z_factor, xy_factor, xy_factor)

    def _resize(arr: np.ndarray, order: int, is_vol: bool = False) -> np.ndarray:
        f = (1.0, *factors) if is_vol and arr.ndim == 4 else factors
        out = zoom(arr, f, order=order)
        t_shape = (arr.shape[0], *shape[-3:]) if is_vol and arr.ndim == 4 else shape[-3:]
        return _center_crop_or_pad(out, t_shape)

    vol = _resize(volume.astype(np.float32), order=1, is_vol=True)
    ci  = np.clip(_resize(gt.cell_indicator, 1), 0, 1).astype(np.float32)
    cc  = _resize(gt.cell_center, 0).astype(np.float32)
    cm  = _resize(gt.cell_mask,   0).astype(np.float32)

    # Displacements scale proportionally with zoom
    mv = np.zeros_like(gt.movement_vectors)
    mv[0] = _resize(gt.movement_vectors[0], 1) * z_factor
    mv[1] = _resize(gt.movement_vectors[1], 1) * xy_factor
    mv[2] = _resize(gt.movement_vectors[2], 1) * xy_factor

    return vol, GroundTruth(ci, cc, cm, mv.astype(np.float32))


def _center_crop_or_pad(arr: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(target, dtype=arr.dtype)
    slices_src: list[slice] = []
    slices_dst: list[slice] = []
    for s, t in zip(arr.shape, target):
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            start = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(start, start + s))
    out[tuple(slices_dst)] = arr[tuple(slices_src)]
    return out


# ---------------------------------------------------------------------------
# Augmentor
# ---------------------------------------------------------------------------

class Augmentor:
    """Apply random augmentations jointly to volume and ground truth."""

    def __init__(self, config: AugmentConfig | None = None, enabled: bool = True):
        self.config  = config or AugmentConfig()
        self.enabled = enabled

    def __call__(
        self,
        volume: np.ndarray,
        gt: GroundTruth,
    ) -> tuple[np.ndarray, GroundTruth]:
        if not self.enabled:
            return volume, gt

        cfg = self.config
        vol, g = volume.copy(), gt

        # Mirror along a random spatial axis (0=Z, 1=Y, 2=X)
        if random.random() < cfg.flip_prob:
            vol, g = _random_flip(vol, g, axis=random.randint(0, 2))

        # Swap Y ↔ X
        if random.random() < cfg.transpose_prob:
            vol, g = _transpose_yx(vol, g)

        # Random spatial shift
        if any(m > 0 for m in cfg.shift_max):
            vol, g = _random_shift(vol, g, cfg.shift_max)

        # Elastic deformation
        if random.random() < cfg.elastic_prob:
            vol, g = _elastic_deform(vol, g, cfg.elastic_alpha, cfg.elastic_sigma)

        # Anisotropic zoom
        if random.random() < cfg.zoom_prob:
            zf  = random.uniform(*cfg.zoom_z_range)
            xyf = random.uniform(*cfg.zoom_xy_range)
            vol, g = _anisotropic_zoom(vol, g, zf, xyf)

        # Intensity augmentation (volume only)
        vol = _intensity_augment(vol, cfg)

        return vol, g
