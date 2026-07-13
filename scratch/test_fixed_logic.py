import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from src.config import Config
from src.models.unet3d import CellTrackingUNet
from src.data.gt_generator import GroundTruth, crop_ground_truth, get_crop_slices
from src.data.augmentations import Augmentor, AugmentConfig

def check():
    cfg = Config()
    cfg.data.input_shape = [48, 192, 192]
    model = CellTrackingUNet.from_config(cfg)
    print("UNet default config:")
    print("  input_shape:", cfg.data.input_shape)
    print("  output_shape:", model.output_shape) # (8, 100, 100)

    # Mock the frame and gt after patch extraction (shape = patch_size)
    patch_size = (48, 192, 192)
    frame = np.random.randn(*patch_size).astype(np.float32)
    gt = GroundTruth(
        cell_indicator=np.random.randn(*patch_size).astype(np.float32),
        cell_center=np.zeros(patch_size, dtype=np.float32),
        cell_mask=np.zeros(patch_size, dtype=np.float32),
        movement_vectors=np.zeros((3, *patch_size), dtype=np.float32)
    )

    # Enable all augmentations to test them (flip, transpose, shift, elastic, zoom)
    aug_cfg = AugmentConfig(
        flip_prob=1.0,
        transpose_prob=1.0,
        elastic_prob=1.0,
        zoom_prob=1.0,
    )
    augmentor = Augmentor(config=aug_cfg, enabled=True)

    # 1. Apply augmentations first (shapes of frame and gt are both patch_size)
    frame_aug, gt_aug = augmentor(frame, gt)
    print("After augmentations:")
    print("  frame shape:", frame_aug.shape)
    print("  gt.cell_indicator shape:", gt_aug.cell_indicator.shape)

    # 2. Crop gt to output_shape at the very end
    output_shape = model.output_shape
    if output_shape is not None:
        crop_slices = get_crop_slices(frame_aug.shape, output_shape)
        gt_aug = crop_ground_truth(gt_aug, crop_slices)

    print("After cropping:")
    print("  frame shape:", frame_aug.shape) # should be (48, 192, 192)
    print("  gt.cell_indicator shape:", gt_aug.cell_indicator.shape) # should be (8, 100, 100)

    # Convert to torch tensors
    volume_tensor = torch.from_numpy(frame_aug).unsqueeze(0).unsqueeze(0).float()
    gt_indicator_tensor = torch.from_numpy(gt_aug.cell_indicator).unsqueeze(0).unsqueeze(0).float()

    # Pass volume to model
    out = model(volume_tensor, inference=False)
    pred_ci = out["cell_indicator"]
    print("Model output cell_indicator shape:", pred_ci.shape) # should be (1, 1, 8, 100, 100)

    # Compute loss using DetectionLoss
    from src.losses.detection_loss import DetectionLoss
    loss_fn = DetectionLoss(output_shape=model.output_shape)
    
    losses = loss_fn(
        {"cell_indicator": pred_ci, "movement_vectors": out["movement_vectors"], "maxima": out["maxima"]},
        {
            "cell_indicator": gt_indicator_tensor,
            "cell_mask": torch.from_numpy(gt_aug.cell_mask).unsqueeze(0).unsqueeze(0).float(),
            "movement_vectors": torch.from_numpy(gt_aug.movement_vectors).unsqueeze(0).float()
        }
    )
    print("Loss computed successfully! Total loss:", losses["loss"].item())

if __name__ == '__main__':
    check()
