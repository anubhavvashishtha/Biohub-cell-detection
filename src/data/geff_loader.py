"""GEFF graph I/O for Biohub ground-truth annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr


@dataclass
class Node:
    node_id: int
    t: int
    z: int
    y: int
    x: int


@dataclass
class TrackingGraph:
    """Parsed GEFF tracking graph."""

    sample_id: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    nodes_by_t: dict[int, list[Node]] = field(default_factory=dict)
    node_by_id: dict[int, Node] = field(default_factory=dict)
    parent_map: dict[int, int] = field(default_factory=dict)
    children_map: dict[int, list[int]] = field(default_factory=dict)
    estimated_node_count: int | None = None

    def nodes_at(self, t: int) -> list[Node]:
        return self.nodes_by_t.get(t, [])


def geff_path_for_zarr(zarr_path: str | Path) -> Path:
    """Return paired .geff path for a .zarr sample."""
    p = Path(zarr_path)
    return p.with_suffix(".geff") if p.suffix == ".zarr" else p.parent / f"{p.name}.geff"


def list_geff_samples(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(".geff"))


def _read_array(group: zarr.Group | zarr.Array, *path: str) -> np.ndarray:
    node = group
    for key in path:
        node = node[key]
    return np.asarray(node[:])


def load_geff(path: str | Path) -> TrackingGraph:
    """Load a .geff directory into a TrackingGraph."""
    path = Path(path)
    sample_id = path.name[:-5] if path.name.endswith(".geff") else path.name
    store = zarr.open(str(path), mode="r")

    node_ids = _read_array(store, "nodes", "ids").astype(np.int64).ravel()
    t_vals = _read_array(store, "nodes", "props", "t", "values").astype(np.int64).ravel()
    z_vals = _read_array(store, "nodes", "props", "z", "values").astype(np.int64).ravel()
    y_vals = _read_array(store, "nodes", "props", "y", "values").astype(np.int64).ravel()
    x_vals = _read_array(store, "nodes", "props", "x", "values").astype(np.int64).ravel()

    edge_ids = _read_array(store, "edges", "ids")
    if edge_ids.ndim == 1:
        edge_ids = edge_ids.reshape(-1, 2)
    edges = [(int(s), int(tgt)) for s, tgt in edge_ids]

    nodes: list[Node] = []
    nodes_by_t: dict[int, list[Node]] = {}
    node_by_id: dict[int, Node] = {}

    for nid, t, z, y, x in zip(node_ids, t_vals, z_vals, y_vals, x_vals, strict=True):
        node = Node(node_id=int(nid), t=int(t), z=int(z), y=int(y), x=int(x))
        nodes.append(node)
        nodes_by_t.setdefault(node.t, []).append(node)
        node_by_id[node.node_id] = node

    parent_map: dict[int, int] = {}
    children_map: dict[int, list[int]] = {}
    for source_id, target_id in edges:
        parent_map[target_id] = source_id
        children_map.setdefault(source_id, []).append(target_id)

    estimated = None
    try:
        import json

        meta_path = path / "zarr.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            estimated = meta.get("estimated_number_of_nodes")
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    return TrackingGraph(
        sample_id=sample_id,
        nodes=nodes,
        edges=edges,
        nodes_by_t=nodes_by_t,
        node_by_id=node_by_id,
        parent_map=parent_map,
        children_map=children_map,
        estimated_node_count=estimated,
    )


def load_paired_sample(zarr_path: str | Path) -> tuple[Any, TrackingGraph | None]:
    """Load zarr volume accessor and optional paired GEFF graph."""
    from .zarr_loader import ZarrVolume

    volume = ZarrVolume(zarr_path)
    geff_path = geff_path_for_zarr(zarr_path)
    graph = load_geff(geff_path) if geff_path.exists() else None
    return volume, graph
