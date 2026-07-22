"""
Smoke tests for the Biohub Cell Tracking implementation.
Run from the project root: python tests/smoke_tests.py
"""
import sys
import traceback

sys.path.insert(0, ".")

PASS = "PASS"
FAIL = "FAIL"


def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"  -> {PASS}")
        return True
    except Exception as e:
        print(f"  -> {FAIL}")
        traceback.print_exc()
        return False


# ─── Test 1: GT generator ────────────────────────────────────────────────────

def test_gt_generator():
    import numpy as np
    from src.data.gt_generator import generate_ground_truth
    from src.data.geff_loader import Node

    nodes = [Node(1, 0, 10, 50, 50), Node(2, 0, 20, 100, 100)]
    gt = generate_ground_truth((64, 256, 256), nodes)

    # cell_indicator should peak at 1.0 at cell centres
    assert gt.cell_indicator.shape == (64, 256, 256), "Wrong shape"
    assert gt.cell_indicator.max() > 0.9, f"Peak too low: {gt.cell_indicator.max()}"
    assert gt.cell_indicator.min() >= 0.0, "Negative values"
    # Exact centres should have value == 1 (outer-product peak)
    assert gt.cell_indicator[10, 50, 50] > 0.9, "Peak not at node centre"
    print(f"  cell_indicator max: {gt.cell_indicator.max():.4f}")
    print(f"  cell_indicator at (10,50,50): {gt.cell_indicator[10,50,50]:.4f}")

    # cell_center should be a point mask
    assert gt.cell_center[10, 50, 50] == 1.0
    assert gt.cell_center[11, 50, 50] == 0.0

    # cell_mask should cover neighbourhood
    assert gt.cell_mask[10, 50, 50] == 1.0

    # movement_vectors: no edges → all zeros
    assert (gt.movement_vectors == 0).all(), "Expected zero vectors (no edges)"


# ─── Test 2: Augmentations ───────────────────────────────────────────────────

def test_augmentations():
    import numpy as np
    from src.data.augmentations import Augmentor, AugmentConfig
    from src.data.gt_generator import GroundTruth

    rng = np.random.default_rng(42)
    vol = rng.uniform(0, 1, (32, 128, 128)).astype(np.float32)
    gt = GroundTruth(
        cell_indicator=rng.uniform(0, 1, (32, 128, 128)).astype(np.float32),
        cell_center=np.zeros((32, 128, 128), dtype=np.float32),
        cell_mask=np.zeros((32, 128, 128), dtype=np.float32),
        movement_vectors=np.zeros((3, 32, 128, 128), dtype=np.float32),
    )
    aug = Augmentor(AugmentConfig(elastic_prob=0.0, zoom_prob=0.0))
    vol2, gt2 = aug(vol, gt)

    assert vol2.shape == vol.shape, f"Shape changed: {vol2.shape}"
    assert gt2.cell_indicator.shape == gt.cell_indicator.shape
    assert gt2.movement_vectors.shape == gt.movement_vectors.shape
    print(f"  volume range after aug: [{vol2.min():.3f}, {vol2.max():.3f}]")


# ─── Test 3: Model forward pass ──────────────────────────────────────────────

def test_model_forward():
    import torch
    from src.config import Config
    from src.models.unet3d import CellTrackingUNet

    cfg = Config()
    m = CellTrackingUNet.from_config(cfg)
    m.eval()

    # Must use the full designed input shape — smaller patches cause valid-padding
    # to collapse intermediate feature maps below the skip connection sizes.
    x = torch.randn(1, 1, 64, 256, 256)
    with torch.no_grad():
        out = m(x, inference=True)

    print(f"  cell_indicator  : {out['cell_indicator'].shape}")
    print(f"  maxima          : {out['maxima'].shape}")
    print(f"  movement_vectors: {out['movement_vectors'].shape}")
    print(f"  output_shape    : {m.output_shape}")

    assert out["cell_indicator"].min() >= 0.0, "cell_indicator < 0"
    assert out["cell_indicator"].max() <= 1.0, "cell_indicator > 1"
    assert out["movement_vectors"].shape[1] == 3, "Expected 3 vector channels"


# ─── Test 4: Residual model ───────────────────────────────────────────────────

def test_residual_model():
    import torch
    from src.config import Config
    from src.models.residual_unet3d import ResidualCellTrackingUNet

    cfg = Config()
    m = ResidualCellTrackingUNet.from_config(cfg)
    m.eval()

    # Must use the full designed input shape
    # ResidualCellTrackingUNet expects a 6D tensor: (B, C, T, Z, Y, X)
    x = torch.randn(1, 1, 7, 64, 256, 256)
    with torch.no_grad():
        out = m(x, inference=True)

    print(f"  ResUNet cell_indicator  : {out['cell_indicator'].shape}")
    print(f"  ResUNet movement_vectors: {out['movement_vectors'].shape}")
    assert out["cell_indicator"].min() >= 0.0


# ─── Test 5: Loss computation ─────────────────────────────────────────────────

def test_loss():
    import torch
    from src.losses.detection_loss import DetectionLoss

    loss_fn = DetectionLoss(lambda_mv=1.0)

    # Synthetic pred/gt with matching shapes
    B, D, H, W = 2, 20, 80, 80
    pred_ci  = torch.sigmoid(torch.randn(B, 1, D, H, W))
    pred_mv  = torch.randn(B, 3, D, H, W)
    gt_ci    = torch.zeros(B, 1, D, H, W)
    gt_ci[0, 0, 10, 40, 40] = 1.0
    gt_mv    = torch.zeros(B, 3, D, H, W)
    cell_mask = torch.zeros(B, 1, D, H, W)
    cell_mask[0, 0, 10, 40, 40] = 1.0

    out = {"cell_indicator": pred_ci, "movement_vectors": pred_mv, "maxima": None}
    tgt = {"cell_indicator": gt_ci,   "movement_vectors": gt_mv,   "cell_mask": cell_mask}

    losses = loss_fn(out, tgt)
    assert "loss"    in losses, "Missing 'loss'"
    assert "loss_ci" in losses, "Missing 'loss_ci'"
    assert "loss_mv" in losses, "Missing 'loss_mv'"
    assert losses["loss"].item() > 0, "Loss should be positive"
    print(f"  total={losses['loss'].item():.4f}  "
          f"ci={losses['loss_ci'].item():.4f}  "
          f"mv={losses['loss_mv'].item():.4f}")


# ─── Test 6: Coordinate offset ───────────────────────────────────────────────

def test_crop_offset():
    from src.infer import compute_crop_offset

    offset = compute_crop_offset((64, 256, 256), (56, 216, 216))
    assert offset == (4, 20, 20), f"Wrong offset: {offset}"
    print(f"  Crop offset: {offset}  CORRECT")


# ─── Test 7: Division-aware linker ───────────────────────────────────────────

def test_linker():
    import numpy as np
    from src.tracking.linker import (
        DetectedNode, link_frames_with_divisions,
        link_frames_hungarian, build_tracking_graph,
        assign_global_node_ids,
    )

    sources = [
        DetectedNode(1, 0, 10, 50, 50, displacement=np.array([0., 0., 0.])),
    ]
    # Two daughters after division
    targets = [
        DetectedNode(2, 1, 10, 55, 50),
        DetectedNode(3, 1, 10, 45, 50),
    ]
    edges = link_frames_with_divisions(
        sources, targets,
        max_distance_um=10.0,
        min_division_distance_um=2.0,
        scale=(1.625, 0.40625, 0.40625),
        use_displacement=False,
    )
    print(f"  Division edges: {edges}")
    assert len(edges) == 2, f"Expected 2 division edges, got {len(edges)}: {edges}"

    # Test standard linker
    edges_h = link_frames_hungarian(sources, targets, max_distance_um=10.0,
                                     scale=(1.625, 0.40625, 0.40625), use_displacement=False)
    assert len(edges_h) == 1, f"Expected 1 edge from Hungarian, got {len(edges_h)}"
    print(f"  Hungarian edges: {edges_h}  (1 as expected)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    results.append(run_test("GT Generator",              test_gt_generator))
    results.append(run_test("Augmentations",             test_augmentations))
    results.append(run_test("Loss Computation",          test_loss))
    results.append(run_test("Crop Offset",               test_crop_offset))
    results.append(run_test("Division-Aware Linker",     test_linker))

    # PyTorch tests — may be skipped if torch is not installed
    try:
        import torch
        results.append(run_test("CellTrackingUNet Forward",   test_model_forward))
        results.append(run_test("ResidualCellTrackingUNet",   test_residual_model))
    except ImportError:
        print("\n[SKIP] PyTorch not installed — skipping model tests")

    print("\n" + "="*60)
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    print("="*60)
    sys.exit(0 if n_fail == 0 else 1)
