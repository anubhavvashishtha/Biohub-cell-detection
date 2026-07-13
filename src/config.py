"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    in_channels: int = 1
    base_features: int = 16
    downsample_factors: list[list[int]] = field(
        default_factory=lambda: [[1, 2, 2], [2, 2, 2], [2, 2, 2]]
    )
    nms_kernel: int = 3
    nms_threshold: float = 0.5


@dataclass
class DataConfig:
    data_root: str = "train"
    patch_size: list[int] = field(default_factory=lambda: [32, 128, 128])
    train_full_volume: bool = False
    gaussian_sigma: list[float] = field(default_factory=lambda: [1.0, 2.0, 2.0])
    cell_mask_radius: int = 2
    scale: list[float] = field(default_factory=lambda: [1.625, 0.40625, 0.40625])
    normalize_mode: str = "per_volume"
    reject_empty: bool = True
    center_on_cell_prob: float = 0.5
    input_shape: list[int] = field(default_factory=lambda: [64, 256, 256])


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    batch_size: int = 2
    epochs: int = 100
    amp: bool = True
    swa_start_epoch: int = 80
    val_every: int = 500
    w_bg: float = 10.0
    w_fg: float = 1.0
    lambda_mv: float = 1.0
    vector_mask_mode: str = "full"
    vector_curriculum_epoch: int = 50
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 4
    val_embryo_fraction: float = 0.2
    seed: int = 42


@dataclass
class InferenceConfig:
    max_link_distance_um: float = 7.0
    batch_size: int = 1
    checkpoint: str = "checkpoints/best.pt"
    output: str = "submission.csv"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        payload = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
