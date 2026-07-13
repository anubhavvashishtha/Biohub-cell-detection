"""Ground-truth generation from sparse cell centroids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from src.data.geff_loader import Node, TrackingGraph


@dataclass
class GroundTruth:
    cell_indicator: np.ndarray   # (Z, Y, X) float32  – Gaussian blobs at cell centres
    cell_center: np.ndarray      # (Z, Y, X) float32  – point-mask at exact centres
    cell_mask: np.ndarray        # (Z, Y, X) float32  – cube/sphere neighbourhood mask
    movement_vectors: np.ndarray # (3, Z, Y, X) float32 – (dz, dy, dx) to parent


# ---------------------------------------------------------------------------
# Gaussian blob rasterisation
# ---------------------------------------------------------------------------

def _place_gaussian_blob(
    volume: np.ndarray,
    center: tuple[int, int, int],
    sigma: tuple[float, float, float],
) -> None:
    """
    Add a normalised separable 3-D Gaussian blob centred at *center* into
    *volume* (in-place).  Uses the outer-product of three 1-D Gaussians, so
    the peak value is always 1.0 before accumulation.
    """
    z, y, x = center
    sz, sy, sx = volume.shape

    # Build each 1-D kernel
    def _kernel_1d(sigma_i: float, size: int, centre_i: int) -> tuple[np.ndarray, int, int]:
        radius = max(1, int(4.0 * sigma_i + 0.5))
        lo = max(0, centre_i - radius)
        hi = min(size, centre_i + radius + 1)
        coords = np.arange(lo, hi, dtype=np.float32) - centre_i
        k = np.exp(-0.5 * (coords / sigma_i) ** 2)
        return k, lo, hi

    kz, z0, z1 = _kernel_1d(sigma[0], sz, z)
    ky, y0, y1 = _kernel_1d(sigma[1], sy, y)
    kx, x0, x1 = _kernel_1d(sigma[2], sx, x)

    # Outer product → 3-D patch (peak = 1.0)
    blob = kz[:, None, None] * ky[None, :, None] * kx[None, None, :]
    volume[z0:z1, y0:y1, x0:x1] = np.maximum(
        volume[z0:z1, y0:y1, x0:x1], blob
    )


def rasterize_cell_indicator(
    shape: tuple[int, int, int],
    nodes: list[Node],
    sigma: tuple[float, float, float] = (1.0, 2.0, 2.0),
) -> np.ndarray:
    """
    Cell-indicator heatmap: max of 3-D Gaussians at annotated centres.

    Values in [0, 1], peak = 1.0 at each labelled cell.
    """
    out = np.zeros(shape, dtype=np.float32)
    for node in nodes:
        if not _in_bounds(node, shape):
            continue
        _place_gaussian_blob(out, (node.z, node.y, node.x), sigma)
    return out


def rasterize_cell_center(
    shape: tuple[int, int, int],
    nodes: list[Node],
) -> np.ndarray:
    """Point mask: 1 exactly at each annotated centre voxel."""
    out = np.zeros(shape, dtype=np.float32)
    for node in nodes:
        if _in_bounds(node, shape):
            out[node.z, node.y, node.x] = 1.0
    return out


def rasterize_cell_mask(
    shape: tuple[int, int, int],
    nodes: list[Node],
    radius: int = 2,
) -> np.ndarray:
    """Binary mask: voxels within Chebyshev *radius* of any cell centre."""
    out = np.zeros(shape, dtype=np.float32)
    for node in nodes:
        if not _in_bounds(node, shape):
            continue
        z0 = max(0, node.z - radius)
        z1 = min(shape[0], node.z + radius + 1)
        y0 = max(0, node.y - radius)
        y1 = min(shape[1], node.y + radius + 1)
        x0 = max(0, node.x - radius)
        x1 = min(shape[2], node.x + radius + 1)
        out[z0:z1, y0:y1, x0:x1] = 1.0
    return out


def sphere_mask_from_centers(
    shape: tuple[int, int, int],
    nodes: list[Node],
    radius_voxels: float = 2.0,
) -> np.ndarray:
    """Euclidean sphere mask (alternative to cube mask)."""
    centers = np.zeros(shape, dtype=bool)
    for node in nodes:
        if _in_bounds(node, shape):
            centers[node.z, node.y, node.x] = True
    if not centers.any():
        return np.zeros(shape, dtype=np.float32)
    dist = distance_transform_edt(~centers)
    return (dist <= radius_voxels).astype(np.float32)


# ---------------------------------------------------------------------------
# Movement-vector rasterisation
# ---------------------------------------------------------------------------

def compute_movement_vectors(
    shape: tuple[int, int, int],
    nodes: list[Node],
    graph: TrackingGraph,
) -> np.ndarray:
    """
    Parent-to-child displacement at each cell location at time t.

    Convention: vector = parent_pos - child_pos  →  (dz, dy, dx) pointing
    *backwards* in time from the child to its parent.  This convention is
    consistent with the inference linker which predicts
    ``next_pos = current_pos - displacement``.
    """
    vectors = np.zeros((3, *shape), dtype=np.float32)
    for node in nodes:
        parent_id = graph.parent_map.get(node.node_id)
        if parent_id is None:
            continue
        parent = graph.node_by_id.get(parent_id)
        if parent is None or parent.t != node.t - 1:
            continue
        if not _in_bounds(node, shape):
            continue
        dz = float(parent.z - node.z)
        dy = float(parent.y - node.y)
        dx = float(parent.x - node.x)
        vectors[0, node.z, node.y, node.x] = dz
        vectors[1, node.z, node.y, node.x] = dy
        vectors[2, node.z, node.y, node.x] = dx
    return vectors


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------

def generate_ground_truth(
    shape: tuple[int, int, int],
    nodes: list[Node],
    graph: TrackingGraph | None = None,
    sigma: tuple[float, float, float] = (1.0, 2.0, 2.0),
    cell_mask_radius: int = 2,
) -> GroundTruth:
    """Generate all GT targets for a single timepoint."""
    graph = graph or TrackingGraph(sample_id="")
    return GroundTruth(
        cell_indicator=rasterize_cell_indicator(shape, nodes, sigma=sigma),
        cell_center=rasterize_cell_center(shape, nodes),
        cell_mask=rasterize_cell_mask(shape, nodes, radius=cell_mask_radius),
        movement_vectors=compute_movement_vectors(shape, nodes, graph),
    )


# ---------------------------------------------------------------------------
# Spatial crop helpers (used by Dataset)
# ---------------------------------------------------------------------------

def crop_ground_truth(gt: GroundTruth, slices: tuple[slice, slice, slice]) -> GroundTruth:
    """Crop all GT arrays with (z, y, x) slices."""
    zsl, ysl, xsl = slices
    return GroundTruth(
        cell_indicator=gt.cell_indicator[zsl, ysl, xsl],
        cell_center=gt.cell_center[zsl, ysl, xsl],
        cell_mask=gt.cell_mask[zsl, ysl, xsl],
        movement_vectors=gt.movement_vectors[:, zsl, ysl, xsl],
    )


def crop_volume(volume: np.ndarray, slices: tuple[slice, slice, slice]) -> np.ndarray:
    zsl, ysl, xsl = slices
    return volume[zsl, ysl, xsl]


def get_crop_slices(
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Center-crop slices to reduce *input_shape* to *output_shape*."""
    offsets = [(i - o) // 2 for i, o in zip(input_shape, output_shape)]
    return tuple(
        slice(off, off + out) for off, out in zip(offsets, output_shape)
    )


def apply_spatial_crop(
    volume: np.ndarray,
    gt: GroundTruth,
    crop_slices: tuple[slice, slice, slice],
) -> tuple[np.ndarray, GroundTruth]:
    return crop_volume(volume, crop_slices), crop_ground_truth(gt, crop_slices)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _in_bounds(node: Node, shape: tuple[int, int, int]) -> bool:
    return (
        0 <= node.z < shape[0]
        and 0 <= node.y < shape[1]
        and 0 <= node.x < shape[2]
    )
