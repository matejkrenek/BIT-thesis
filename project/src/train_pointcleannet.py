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
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dotenv import load_dotenv
from pytorch3d.ops import sample_farthest_points
import open3d as o3d
import matplotlib.pyplot as plt

import safe_gpu


from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import (
    Combined,
    Noise,
    OutlierPoints,
    FloatingCluster,
    HairLikeNoise,
    SurfaceBridgingArtifact,
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
NUM_GPUS = torch.cuda.device_count()
print(f"[INFO] Available GPUs: {NUM_GPUS}")

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = Path(DATA_FOLDER_PATH) / "data" / "ShapeNetV2"
CHECKPOINT_DIR = Path(DATA_FOLDER_PATH) / "checkpoints" / "pointcleannet"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 120
SAVE_EVERY = 10
RESUME_FROM: Optional[str] = None
OVERFIT = True

W_CHAMFER = 1.0
W_CONSISTENCY = 0.02
W_SMOOTH = 0.01

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
    query_chunk_size = 1024
    use_point_stn = True
    use_feat_stn = True


# ============================================================================
# LOGGER / NOTIFIER SETUP
# ============================================================================

notifier = DiscordNotifier(
    webhook_url=os.getenv(
        "DISCORD_WEBHOOK_URL",
        "https://discord.com/api/webhooks/1466392738609238046/YOGa8j4HL9wKYeQXXyFdIR_j-vxs5jGYYekNnY0YSlBy-0aJnFwHXMfGPNxxLkMh5FE-",
    ),
    project_name="BIT Thesis Project - PointCleanNet",
    project_url="https://github.com/matejkrenek/BIT-thesis",
    avatar_name="PointCleanNet Training Bot",
)


# ============================================================================
# DATASET AND DATALOADERS
# ============================================================================


def create_dataset(defect_augmentation_count: int = 6):
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
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    if OVERFIT:
        dataset = Subset(dataset, list(range(batch_size * 2)))

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
        batch_size=batch_size if not OVERFIT else max(1, batch_size // 2),
        shuffle=not OVERFIT,
        num_workers=4,
        persistent_workers=True,
        collate_fn=pointcleannet_collate,
        pin_memory=True,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pointcleannet_collate,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for originals, padded, lengths in train_loader:
        if originals is None or padded is None or lengths is None:
            continue

        originals = originals.to(device, non_blocking=True)
        padded = padded.to(device, non_blocking=True)
        lengths = lengths.to(device)

        defecteds, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )

        pred = model(defecteds)

        if hasattr(model, "module"):
            loss = model.module.compute_loss(
                pred=pred,
                target=originals,
                input_points=defecteds,
                w_chamfer=W_CHAMFER,
                w_consistency=W_CONSISTENCY,
                w_smooth=W_SMOOTH,
            )
        else:
            loss = model.compute_loss(
                pred=pred,
                target=originals,
                input_points=defecteds,
                w_chamfer=W_CHAMFER,
                w_consistency=W_CONSISTENCY,
                w_smooth=W_SMOOTH,
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for originals, padded, lengths in val_loader:
        if originals is None or padded is None or lengths is None:
            continue

        originals = originals.to(device, non_blocking=True)
        padded = padded.to(device, non_blocking=True)
        lengths = lengths.to(device)

        defecteds, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )

        pred = model(defecteds)

        if hasattr(model, "module"):
            loss = model.module.compute_loss(
                pred=pred,
                target=originals,
                input_points=defecteds,
                w_chamfer=W_CHAMFER,
                w_consistency=W_CONSISTENCY,
                w_smooth=W_SMOOTH,
            )
        else:
            loss = model.compute_loss(
                pred=pred,
                target=originals,
                input_points=defecteds,
                w_chamfer=W_CHAMFER,
                w_consistency=W_CONSISTENCY,
                w_smooth=W_SMOOTH,
            )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: StepLR,
    epoch: int,
    val_loss: float,
    checkpoint_path: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


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


def save_loss_plot(train_losses: list, val_losses: list, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("PointCleanNet Denoising Training Progress", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def main():
    print(f"[INFO] Training on {DEVICE}")
    print(f"[INFO] Number of GPUs: {NUM_GPUS}")

    print("[INFO] Creating dataset...")
    dataset = create_dataset()
    train_loader, val_loader, test_loader = create_data_loaders(dataset)

    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Train size: {len(train_loader.dataset)}")
    print(f"[INFO] Val size: {len(val_loader.dataset)}")
    print(f"[INFO] Test size: {len(test_loader.dataset)}")

    config = PointCleanNetConfig()
    model = create_model(config)
    print(
        "[INFO] Model config: "
        f"k={config.k_neighbors}, local_dim={config.local_feature_dim}, "
        f"hidden_dim={config.hidden_dim}, stages={config.num_stages}"
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

    start_epoch = 1
    best_val_loss = float("inf")

    if RESUME_FROM is not None and Path(RESUME_FROM).exists():
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, RESUME_FROM, DEVICE
        )

    train_losses = []
    val_losses = []

    if RESUME_FROM is None:
        notifier.send_training_start(
            total_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            train_size=len(train_loader.dataset),
            val_size=len(val_loader.dataset),
            training_on=DEVICE,
            number_of_gpus=NUM_GPUS,
            learning_rate=LEARNING_RATE,
        )

    print(f"[INFO] Starting training for {EPOCHS} epochs")
    start_time = time.time()

    try:
        epoch_pbar = tqdm(
            range(start_epoch, EPOCHS + 1),
            desc="Training",
            unit="epoch",
        )

        for epoch in epoch_pbar:
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
            val_loss = validate(model, val_loader, DEVICE)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step()

            elapsed = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            epochs_left = EPOCHS - epoch
            avg_epoch_time = elapsed / max(epochs_done, 1)
            eta_seconds = int(avg_epoch_time * epochs_left)

            epoch_pbar.set_postfix(
                train=f"{train_loss:.6f}",
                val=f"{val_loss:.6f}",
                best=f"{best_val_loss:.6f}",
                eta=f"{eta_seconds // 60}m {eta_seconds % 60}s",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = CHECKPOINT_DIR / "v1_best.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                    best_checkpoint_path,
                )
                print(f"[SAVED] Best model at epoch {epoch} with loss {val_loss:.6f}")

            if epoch % SAVE_EVERY == 0:
                periodic_checkpoint_path = (
                    CHECKPOINT_DIR / f"v1_checkpoint_epoch_{epoch:04d}.pt"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                    periodic_checkpoint_path,
                )

            loss_plot_path = CHECKPOINT_DIR / "v1_loss_curve.png"
            save_loss_plot(train_losses, val_losses, loss_plot_path)

            current_lr = scheduler.get_last_lr()[0]
            notifier.send_training_progress(
                epoch=epoch,
                total_epochs=EPOCHS,
                current_loss=val_loss,
                best_loss=best_val_loss,
                learning_rate=current_lr,
                batch_size=BATCH_SIZE,
                elapsed_time=f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
                estimated_finish_time=time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta_seconds)
                ),
                loss_curve_path=str(loss_plot_path),
            )

        total_time = time.time() - start_time
        print(
            f"[INFO] Training completed in {int(total_time // 3600)}h "
            f"{int((total_time % 3600) // 60)}m {int(total_time % 60)}s"
        )

        final_plot_path = CHECKPOINT_DIR / "v1_final_loss_curve.png"
        save_loss_plot(train_losses, val_losses, final_plot_path)

        notifier.send_training_completion(
            total_epochs=EPOCHS,
            final_loss=val_losses[-1],
            best_loss=best_val_loss,
            training_time=f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s",
            final_loss_curve_path=str(final_plot_path),
            best_model_path=str(CHECKPOINT_DIR / "v1_best.pt"),
        )

    except Exception as e:
        print(f"[ERROR] Training interrupted with error: {e}")
        notifier.send_training_error(
            error_message=str(e),
            epoch=epoch if "epoch" in locals() else -1,
        )
        raise


if __name__ == "__main__":
    main()
