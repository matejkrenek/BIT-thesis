"""
PointCleanNet-style Denoiser Training Script

Training script aligned with train_pointr.py structure, adapted for
point-cloud denoising on augmented ShapeNet data.
"""

import os
import time
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dotenv import load_dotenv
from pytorch3d.ops import sample_farthest_points
import open3d as o3d
import matplotlib.pyplot as plt
from visualize.utils import plot_pointcloud_to_image
import safe_gpu


from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import (
    Combined,
    Noise,
    OutlierPoints,
    FloatingCluster,
    HairLikeNoise,
    SurfaceBridgingArtifact,
    SurfaceToPlaneBridge,
    SurfaceFlattening,
)
from models import PointCleanNetDenoiser
from notifications import DiscordNotifier


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
safe_gpu.claim_gpus(1)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = 1
print(f"[INFO] Available GPUs: {NUM_GPUS}")

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = Path(DATA_FOLDER_PATH) / "data" / "ShapeNetV2"
CHECKPOINT_DIR = Path(DATA_FOLDER_PATH) / "checkpoints" / "pointcleannet"
CHECKPOINT = CHECKPOINT_DIR / "v1_best.pt"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)


class PointCleanNetConfig:
    k_neighbors = 32
    local_feature_dim = 128
    hidden_dim = 256
    num_stages = 2
    max_offset = 0.05
    query_chunk_size = 256
    use_point_stn = True
    use_feat_stn = True


# ============================================================================
# DATASET AND DATALOADERS
# ============================================================================


def create_dataset(defect_augmentation_count: int = 2):
    """Create augmented ShapeNet dataset with photogrammetry-like denoising defects."""
    rng = np.random.RandomState(SEED)

    dataset = AugmentedDataset(
        dataset=ShapeNetDataset(root=str(ROOT_DATA)),
        defects=[
            Combined(
                [
                    Noise(sigma=rng.uniform(0.001, 0.008)),
                    OutlierPoints(
                        num_points=int(rng.randint(40, 220)),
                        scale_factor=rng.uniform(1.2, 2.2),
                        mode="uniform",
                    ),
                    FloatingCluster(
                        cluster_radius=rng.uniform(0.01, 0.06),
                        clusters=int(rng.randint(1, 4)),
                        points_per_cluster=int(rng.randint(20, 120)),
                        offset_factor=rng.uniform(1.1, 2.2),
                    ),
                    HairLikeNoise(
                        num_spikes=int(rng.randint(15, 80)),
                        max_length=rng.uniform(0.02, 0.08),
                        jitter=rng.uniform(0.001, 0.004),
                    ),
                    SurfaceBridgingArtifact(
                        num_surfaces=int(rng.randint(1, 3)),
                        resolution_u=int(rng.randint(15, 35)),
                        resolution_v=int(rng.randint(4, 9)),
                        width=rng.uniform(0.008, 0.03),
                        jitter=rng.uniform(0.001, 0.006),
                    ),
                    SurfaceToPlaneBridge(
                        num_bridges=int(rng.randint(6, 16)),
                        points_per_bridge=int(rng.randint(10, 22)),
                        plane_offset_ratio=rng.uniform(0.02, 0.08),
                        axis=1,
                        bottom_band_ratio=rng.uniform(0.25, 0.55),
                        lateral_jitter=rng.uniform(0.0008, 0.003),
                        normal_jitter=rng.uniform(0.0005, 0.0025),
                    ),
                ]
            )
            for _ in range(defect_augmentation_count)
        ],
    )

    return dataset


def pointcleannet_collate(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for variable-size defected clouds."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None

    originals, defecteds = zip(*batch)
    originals = torch.stack(originals, dim=0)

    lengths = torch.tensor([pc.shape[0] for pc in defecteds], dtype=torch.long)
    max_n = lengths.max().item()

    padded = torch.zeros(len(defecteds), max_n, 3)
    for i, pc in enumerate(defecteds):
        padded[i, : pc.shape[0]] = pc

    return originals, padded, lengths


def create_data_loaders(
    dataset,
    train_size: float = 0.8,
    val_size: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    actual_train_size = int(train_size * len(dataset))
    actual_val_size = int(val_size * len(dataset))
    actual_test_size = len(dataset) - actual_train_size - actual_val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [actual_train_size, actual_val_size, actual_test_size],
        generator=g,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=pointcleannet_collate,
        pin_memory=True,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=pointcleannet_collate,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=pointcleannet_collate,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# MODEL SETUP
# ============================================================================


def create_model(config: PointCleanNetConfig) -> nn.Module:
    model = PointCleanNetDenoiser(
        k_neighbors=config.k_neighbors,
        local_feature_dim=config.local_feature_dim,
        hidden_dim=config.hidden_dim,
        num_stages=config.num_stages,
        max_offset=config.max_offset,
        query_chunk_size=config.query_chunk_size,
        use_point_stn=config.use_point_stn,
        use_feat_stn=config.use_feat_stn,
    )

    if DEVICE == "cuda" and NUM_GPUS > 1:
        print("[INFO] Using DataParallel with multiple GPUs")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)
    return model


def load_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: StepLR,
    checkpoint_path: str,
    device: str,
) -> Tuple[int, float]:
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint["epoch"] + 1, checkpoint["val_loss"]


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def main():
    dataset = create_dataset()
    train_loader, val_loader, test_loader = create_data_loaders(dataset)

    config = PointCleanNetConfig()
    model = create_model(config)
    model.eval()  # Set to eval mode for inference

    for originals, padded, lengths in train_loader:
        if originals is None or padded is None or lengths is None:
            continue

        originals = originals.to(DEVICE, non_blocking=True)
        padded = padded.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)

        defecteds, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
        )

        pred = model(defecteds)

        plot_pointcloud_to_image(originals[0], "pcd_original_0.png")
        plot_pointcloud_to_image(defecteds[0], "pcd_defected_0.png")
        plot_pointcloud_to_image(pred[0], "pcd_completed_0.png")

        break


if __name__ == "__main__":
    main()
