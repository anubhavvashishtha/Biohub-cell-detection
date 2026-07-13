"""Zarr v3 volume I/O for Biohub microscopy data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import zarr


def list_zarr_samples(data_root: str | Path) -> list[Path]:
    """List all .zarr sample directories under data_root."""
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(".zarr"))


def sample_id_from_path(path: str | Path) -> str:
    """Return dataset id (folder name without .zarr extension)."""
    name = Path(path).name
    return name[:-5] if name.endswith(".zarr") else name


def embryo_id_from_sample(sample_id: str) -> str:
    """First segment of folder name identifies the embryo."""
    return sample_id.split("_")[0]


def open_volume(path: str | Path) -> zarr.Array:
    """Open the main array at path 0/ inside a .zarr directory."""
    store = zarr.open(str(path), mode="r")
    if "0" in store:
        return store["0"]
    if isinstance(store, zarr.Array):
        return store
    raise KeyError(f"No array at '0/' in {path}")


def read_volume_metadata(path: str | Path) -> dict:
    """Read shape, dtype, and chunk info from zarr metadata."""
    arr = open_volume(path)
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "chunks": arr.chunks,
    }


def read_timepoint(path: str | Path, t: int) -> np.ndarray:
    """Read a single timepoint as (Z, Y, X) uint16 array."""
    arr = open_volume(path)
    return np.asarray(arr[t], dtype=np.uint16)


def read_volume(
    path: str | Path,
    timepoints: slice | list[int] | None = None,
) -> np.ndarray:
    """Read volume or subset of timepoints as (T, Z, Y, X)."""
    arr = open_volume(path)
    if timepoints is None:
        return np.asarray(arr[:], dtype=np.uint16)
    if isinstance(timepoints, slice):
        return np.asarray(arr[timepoints], dtype=np.uint16)
    return np.asarray([arr[t] for t in timepoints], dtype=np.uint16)


def normalize_volume(
    volume: np.ndarray,
    mode: Literal["per_volume", "percentile"] = "per_volume",
    percentile: float = 99.5,
) -> np.ndarray:
    """Normalize uint16 volume to float32 in [0, 1]."""
    vol = volume.astype(np.float32)
    if mode == "per_volume":
        vmax = float(vol.max())
        if vmax > 0:
            vol /= vmax
    elif mode == "percentile":
        vmax = float(np.percentile(vol, percentile))
        if vmax > 0:
            vol = np.clip(vol / vmax, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    return vol


class ZarrVolume:
    """Lazy accessor for a single .zarr sample."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.sample_id = sample_id_from_path(self.path)
        self.embryo_id = embryo_id_from_sample(self.sample_id)
        self._array = open_volume(self.path)
        self.shape = self._array.shape  # (T, Z, Y, X)
        self.num_timepoints = self.shape[0]

    def __len__(self) -> int:
        return self.num_timepoints

    def get_frame(self, t: int) -> np.ndarray:
        return np.asarray(self._array[t], dtype=np.uint16)

    def get_frame_normalized(
        self,
        t: int,
        mode: Literal["per_volume", "percentile"] = "per_volume",
    ) -> np.ndarray:
        return normalize_volume(self.get_frame(t), mode=mode)
