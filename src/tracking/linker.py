"""Frame-to-frame cell linking via displacement and Hungarian assignment.

Supports:
  - Standard 1-to-1 linking (Hungarian algorithm).
  - Division-aware linking: one source cell may link to up to two target
    cells (cell division / mitosis), identified by proximity to the
    predicted displacement vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedNode:
    node_id: int
    t: int
    z: int
    y: int
    x: int
    confidence: float = 1.0
    # Predicted (dz, dy, dx) to parent: next_pos = pos - displacement
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass
class TrackingResult:
    nodes: list[DetectedNode]
    edges: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

SCALE_DEFAULT = (1.625, 0.40625, 0.40625)


def physical_distance(
    pos_a: tuple[float, float, float],
    pos_b: tuple[float, float, float],
    scale: Sequence[float] = SCALE_DEFAULT,
) -> float:
    dz = (pos_a[0] - pos_b[0]) * scale[0]
    dy = (pos_a[1] - pos_b[1]) * scale[1]
    dx = (pos_a[2] - pos_b[2]) * scale[2]
    return float(np.sqrt(dz * dz + dy * dy + dx * dx))


def predict_next_position(node: DetectedNode) -> tuple[float, float, float]:
    """
    Predict forward position using stored displacement.

    GT stores displacement = parent_pos - child_pos, i.e. pointing *back*.
    To predict where the child will be next frame:
        next_pos = current_pos - displacement
    """
    return (
        node.z - node.displacement[0],
        node.y - node.displacement[1],
        node.x - node.displacement[2],
    )


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------

def extract_peaks_from_frame(
    cell_indicator: np.ndarray,
    maxima: np.ndarray,
    movement_vectors: np.ndarray,
    t: int,
    threshold: float = 0.5,
    start_node_id: int = 1,
    crop_offset: tuple[int, int, int] = (0, 0, 0),
) -> list[DetectedNode]:
    """
    Extract detected nodes from NMS maxima in one frame.

    Args:
        cell_indicator : (D, H, W) or (1, D, H, W) sigmoid activations.
        maxima         : boolean (D, H, W) or (1, D, H, W) — NMS local maxima.
        movement_vectors: (3, D, H, W) predicted (dz, dy, dx) displacement.
        t              : timepoint index.
        threshold      : minimum cell_indicator value to accept a peak.
        start_node_id  : first node_id to assign (local frame IDs; reassigned later).
        crop_offset    : (dz, dy, dx) voxel offset to add to all coordinates so
                         they are expressed in the *full-volume* coordinate frame
                         rather than the valid-padded network output frame.
    """
    if cell_indicator.ndim == 4:
        cell_indicator = cell_indicator[0]
    if maxima.ndim == 4:
        maxima = maxima[0]

    oz, oy, ox = crop_offset
    coords = np.argwhere(maxima & (cell_indicator > threshold))
    nodes: list[DetectedNode] = []
    for i, (z, y, x) in enumerate(coords):
        disp = movement_vectors[:, z, y, x].astype(np.float32)
        nodes.append(
            DetectedNode(
                node_id=start_node_id + i,
                t=t,
                z=int(z) + oz,
                y=int(y) + oy,
                x=int(x) + ox,
                confidence=float(cell_indicator[z, y, x]),
                displacement=disp,
            )
        )
    return nodes


# ---------------------------------------------------------------------------
# Standard 1-to-1 linker (Hungarian)
# ---------------------------------------------------------------------------

def link_frames_hungarian(
    sources: list[DetectedNode],
    targets: list[DetectedNode],
    max_distance_um: float = 7.0,
    scale: Sequence[float] = SCALE_DEFAULT,
    use_displacement: bool = True,
) -> list[tuple[int, int]]:
    """
    Link cells from frame t (sources) to frame t+1 (targets) — 1-to-1.

    Returns list of (source_node_id, target_node_id) edges.
    """
    if not sources or not targets:
        return []

    n_src = len(sources)
    n_tgt = len(targets)
    INF = max_distance_um * 10.0
    cost = np.full((n_src, n_tgt), fill_value=INF, dtype=np.float64)

    for i, src in enumerate(sources):
        ref = predict_next_position(src) if use_displacement else (float(src.z), float(src.y), float(src.x))
        for j, tgt in enumerate(targets):
            d = physical_distance(ref, (float(tgt.z), float(tgt.y), float(tgt.x)), scale)
            if d <= max_distance_um:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (sources[r].node_id, targets[c].node_id)
        for r, c in zip(row_ind, col_ind)
        if cost[r, c] <= max_distance_um
    ]


# ---------------------------------------------------------------------------
# Division-aware linker (1→2 splits)
# ---------------------------------------------------------------------------

def link_frames_with_divisions(
    sources: list[DetectedNode],
    targets: list[DetectedNode],
    max_distance_um: float = 7.0,
    min_division_distance_um: float = 2.0,
    scale: Sequence[float] = SCALE_DEFAULT,
    use_displacement: bool = True,
) -> list[tuple[int, int]]:
    """
    Link cells across frames allowing one source → up to 2 targets (division).

    Algorithm:
      1. Build a cost matrix (sources × targets).
      2. Run Hungarian assignment for the primary (1-to-1) links.
      3. For every *unmatched* target that is within max_distance_um of an
         already-matched source, propose a division edge — but only if the
         two daughters are at least *min_division_distance_um* apart from
         each other (to avoid duplicate detections).

    Returns a list of (source_node_id, target_node_id) edges.
    """
    if not sources or not targets:
        return []

    # ---- Step 1: build distance matrix ----
    INF = max_distance_um * 10.0
    n_src = len(sources)
    n_tgt = len(targets)
    dist = np.full((n_src, n_tgt), fill_value=INF, dtype=np.float64)

    for i, src in enumerate(sources):
        ref = (
            predict_next_position(src)
            if use_displacement
            else (float(src.z), float(src.y), float(src.x))
        )
        for j, tgt in enumerate(targets):
            d = physical_distance(ref, (float(tgt.z), float(tgt.y), float(tgt.x)), scale)
            if d <= max_distance_um:
                dist[i, j] = d

    # ---- Step 2: primary 1-to-1 assignment ----
    row_ind, col_ind = linear_sum_assignment(dist)
    primary_edges: list[tuple[int, int]] = []
    matched_src_idx: set[int] = set()
    matched_tgt_idx: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if dist[r, c] <= max_distance_um:
            primary_edges.append((sources[r].node_id, targets[c].node_id))
            matched_src_idx.add(r)
            matched_tgt_idx.add(c)

    # ---- Step 3: division — unmatched targets near matched sources ----
    division_edges: list[tuple[int, int]] = []
    unmatched_tgt = [j for j in range(n_tgt) if j not in matched_tgt_idx]

    for j in unmatched_tgt:
        tgt_j = targets[j]
        tgt_pos = (float(tgt_j.z), float(tgt_j.y), float(tgt_j.x))

        # Find the closest matched source for this unmatched target
        best_d, best_i = INF, -1
        for i in matched_src_idx:
            d = dist[i, j]
            if d < best_d:
                best_d, best_i = d, i

        if best_i < 0 or best_d > max_distance_um:
            continue  # too far away

        # Find the sibling (primary target of the matched source)
        sibling_edge = next(
            (e for e in primary_edges if e[0] == sources[best_i].node_id), None
        )
        if sibling_edge is None:
            continue
        sibling_id = sibling_edge[1]
        sibling = next((t for t in targets if t.node_id == sibling_id), None)
        if sibling is None:
            continue

        # Sisters must be sufficiently far apart to avoid double-detections
        sib_pos = (float(sibling.z), float(sibling.y), float(sibling.x))
        if physical_distance(tgt_pos, sib_pos, scale) < min_division_distance_um:
            continue

        division_edges.append((sources[best_i].node_id, tgt_j.node_id))

    return primary_edges + division_edges


# ---------------------------------------------------------------------------
# Full graph builder
# ---------------------------------------------------------------------------

def build_tracking_graph(
    frame_detections: dict[int, list[DetectedNode]],
    max_distance_um: float = 7.0,
    scale: Sequence[float] = SCALE_DEFAULT,
    allow_divisions: bool = True,
    min_division_distance_um: float = 2.0,
) -> TrackingResult:
    """
    Link detections across all frames into a tracking graph.

    Args:
        frame_detections      : {timepoint: [DetectedNode, …]}
        max_distance_um       : maximum physical distance (µm) for any link.
        scale                 : physical voxel scale (z, y, x) µm/voxel.
        allow_divisions       : if True, use division-aware linker.
        min_division_distance_um: minimum distance between two daughters.
    """
    all_nodes: list[DetectedNode] = []
    for nodes in frame_detections.values():
        all_nodes.extend(nodes)

    timepoints  = sorted(frame_detections.keys())
    all_edges: list[tuple[int, int]] = []

    linker = link_frames_with_divisions if allow_divisions else link_frames_hungarian

    for t0, t1 in zip(timepoints[:-1], timepoints[1:]):
        if t1 != t0 + 1:
            continue
        edges = linker(
            frame_detections[t0],
            frame_detections[t1],
            max_distance_um=max_distance_um,
            scale=scale,
            **({"min_division_distance_um": min_division_distance_um} if allow_divisions else {}),
        )
        all_edges.extend(edges)

    return TrackingResult(nodes=all_nodes, edges=all_edges)


# ---------------------------------------------------------------------------
# Node ID management
# ---------------------------------------------------------------------------

def assign_global_node_ids(
    frame_detections: dict[int, list[DetectedNode]],
) -> dict[int, list[DetectedNode]]:
    """Reassign consecutive node IDs per dataset (1-indexed)."""
    next_id = 1
    out: dict[int, list[DetectedNode]] = {}
    for t in sorted(frame_detections.keys()):
        nodes: list[DetectedNode] = []
        for node in frame_detections[t]:
            nodes.append(
                DetectedNode(
                    node_id=next_id,
                    t=node.t,
                    z=node.z,
                    y=node.y,
                    x=node.x,
                    confidence=node.confidence,
                    displacement=node.displacement.copy(),
                )
            )
            next_id += 1
        out[t] = nodes
    return out
