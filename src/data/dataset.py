"""PyTorch Dataset for Biohub cell detection training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import Config
from src.data.augmentations import Augmentor
from src.data.geff_loader import TrackingGraph, load_geff
from src.data.gt_generator import (
    GroundTruth,
    crop_ground_truth,
    crop_volume,
    generate_ground_truth,
    get_crop_slices,
)
from src.data.zarr_loader import ZarrVolume, embryo_id_from_sample, list_zarr_samples, normalize_volume


@dataclass
class SampleIndex:
    sample_path: Path
    timepoint: int
    weight: float


def _extract_patch_slices(
    shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    center: tuple[int, int, int] | None = None,
) -> tuple[slice, slice, slice]:
    pz, py, px = patch_size
    sz, sy, sx = shape

    if center is None:
        z0 = random.randint(0, max(0, sz - pz))
        y0 = random.randint(0, max(0, sy - py))
        x0 = random.randint(0, max(0, sx - px))
    else:
        cz, cy, cx = center
        z0 = int(np.clip(cz - pz // 2, 0, max(0, sz - pz)))
        y0 = int(np.clip(cy - py // 2, 0, max(0, sy - py)))
        x0 = int(np.clip(cx - px // 2, 0, max(0, sx - px)))

    return slice(z0, z0 + pz), slice(y0, y0 + py), slice(x0, x0 + px)


def _crop_with_slices(volume: np.ndarray, gt: GroundTruth, slices: tuple[slice, slice, slice]):
    zsl, ysl, xsl = slices
    vol = volume[:, zsl, ysl, xsl] if volume.ndim == 4 else crop_volume(volume, slices)
    return (
        vol,
        GroundTruth(
            cell_indicator=gt.cell_indicator[zsl, ysl, xsl],
            cell_center=gt.cell_center[zsl, ysl, xsl],
            cell_mask=gt.cell_mask[zsl, ysl, xsl],
            movement_vectors=gt.movement_vectors[:, zsl, ysl, xsl],
        ),
    )


class BiohubDetectionDataset(Dataset):
    """
    Yields (volume, gt_dict, meta) for one timepoint.

    gt_dict keys: cell_indicator, cell_center, cell_mask, movement_vectors (all tensors).
    """

    def __init__(
        self,
        config: Config,
        sample_paths: list[Path] | None = None,
        augment: bool = True,
        output_shape: tuple[int, int, int] | None = None,
    ):
        self.config = config
        self.data_cfg = config.data
        self.augment = augment
        self.output_shape = output_shape

        root = Path(self.data_cfg.data_root)
        self.sample_paths = sample_paths or list_zarr_samples(root)
        self.volumes: list[ZarrVolume] = [ZarrVolume(p) for p in self.sample_paths]
        self.graphs: list[TrackingGraph | None] = []
        for p in self.sample_paths:
            geff = p.with_suffix(".geff")
            self.graphs.append(load_geff(geff) if geff.exists() else None)

        self.indices: list[SampleIndex] = self._build_indices()
        self.augmentor = Augmentor(enabled=augment)

    def _build_indices(self) -> list[SampleIndex]:
        indices: list[SampleIndex] = []
        for vol, graph, path in zip(self.volumes, self.graphs, self.sample_paths):
            if graph is None:
                continue
            for t in range(vol.num_timepoints):
                nodes = graph.nodes_at(t)
                if self.data_cfg.reject_empty and len(nodes) == 0:
                    continue
                weight = max(1.0, float(len(nodes)))
                indices.append(SampleIndex(path, t, weight))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def _weighted_choice_index(self) -> int:
        weights = [idx.weight for idx in self.indices]
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                return i
        return len(weights) - 1

    def __getitem__(self, index: int) -> dict:
        if self.augment:
            index = self._weighted_choice_index()

        entry = self.indices[index]
        vol_idx = next(i for i, p in enumerate(self.sample_paths) if p == entry.sample_path)
        volume = self.volumes[vol_idx]
        graph = self.graphs[vol_idx]
        assert graph is not None

        t = entry.timepoint
        
        frames = []
        for i in range(-3, 4):
            # Duplicate boundary frames (e.g. at t=0, it will use t=0 for i=-3, -2, -1)
            t_offset = max(0, min(t + i, volume.num_timepoints - 1))
            f = normalize_volume(volume.get_frame(t_offset), mode=self.data_cfg.normalize_mode)
            frames.append(f)
            
        frame = np.stack(frames, axis=0) # Shape: (7, Z, Y, X)
        
        nodes = graph.nodes_at(t)
        full_shape = frame.shape[1:] # Spatial shape
        gt = generate_ground_truth(
            full_shape,
            nodes,
            graph=graph,
            sigma=tuple(self.data_cfg.gaussian_sigma),
            cell_mask_radius=self.data_cfg.cell_mask_radius,
        )

        if self.data_cfg.train_full_volume:
            patch_slices = (slice(0, full_shape[0]), slice(0, full_shape[1]), slice(0, full_shape[2]))
        else:
            center = None
            if nodes and random.random() < self.data_cfg.center_on_cell_prob:
                node = random.choice(nodes)
                center = (node.z, node.y, node.x)
            patch_size = tuple(self.data_cfg.patch_size)
            patch_slices = _extract_patch_slices(full_shape, patch_size, center=center)

        frame, gt = _crop_with_slices(frame, gt, patch_slices)

        if self.augment:
            frame, gt = self.augmentor(frame, gt)

        if self.output_shape is not None:
            crop_slices = get_crop_slices(frame.shape[1:], self.output_shape)
            gt = crop_ground_truth(gt, crop_slices)

        return {
            "volume": torch.from_numpy(frame).unsqueeze(0).float(),
            "cell_indicator": torch.from_numpy(gt.cell_indicator).unsqueeze(0).float(),
            "cell_center": torch.from_numpy(gt.cell_center).unsqueeze(0).float(),
            "cell_mask": torch.from_numpy(gt.cell_mask).unsqueeze(0).float(),
            "movement_vectors": torch.from_numpy(gt.movement_vectors).float(),
            "meta": {
                "sample_id": volume.sample_id,
                "embryo_id": embryo_id_from_sample(volume.sample_id),
                "timepoint": t,
            },
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "volume": torch.stack([b["volume"] for b in batch]),
        "cell_indicator": torch.stack([b["cell_indicator"] for b in batch]),
        "cell_center": torch.stack([b["cell_center"] for b in batch]),
        "cell_mask": torch.stack([b["cell_mask"] for b in batch]),
        "movement_vectors": torch.stack([b["movement_vectors"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def split_samples_by_embryo(
    sample_paths: list[Path],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Embryo-disjoint train/val split."""
    rng = random.Random(seed)
    by_embryo: dict[str, list[Path]] = {}
    for p in sample_paths:
        eid = embryo_id_from_sample(p.name.replace(".zarr", ""))
        by_embryo.setdefault(eid, []).append(p)

    embryo_ids = sorted(by_embryo.keys())
    rng.shuffle(embryo_ids)
    n_val = max(1, int(len(embryo_ids) * val_fraction))
    val_embryos = set(embryo_ids[:n_val])

    train_paths, val_paths = [], []
    for eid, paths in by_embryo.items():
        (val_paths if eid in val_embryos else train_paths).extend(paths)

    # Ensure training set has more samples than validation set
    if len(train_paths) < len(val_paths):
        train_paths, val_paths = val_paths, train_paths

    return train_paths, val_paths
