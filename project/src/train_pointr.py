"""
PoinTr Training Script

Improved and well-structured training script for the PoinTr point cloud completion model.
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
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dotenv import load_dotenv
import safe_gpu
from pytorch3d.ops import sample_farthest_points
import open3d as o3d
import matplotlib.pyplot as plt

from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import LargeMissingRegion, LocalDropout, Noise, Combined
from models.pointr import PoinTr
from notifications import DiscordNotifier


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
safe_gpu.claim_gpus(1)

# Suppress open3d warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# Device and GPU setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
print(f"[INFO] Available GPUs: {NUM_GPUS}")

# Paths
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = Path(DATA_FOLDER_PATH) / "data" / "ShapeNetV2"
CHECKPOINT_DIR = Path(DATA_FOLDER_PATH) / "checkpoints" / "pointr"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
SAVE_EVERY = 10
RESUME_FROM: Optional[str] = None
OVERFIT = True

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)

# PoinTr specific configuration
class PoinTrConfig:
    trans_dim = 384
    knn_layer = 0
    num_pred = 16384
    num_query = 128


# ============================================================================
# LOGGER / NOTIFIER SETUP
# ============================================================================

notifier = DiscordNotifier(
    webhook_url=os.getenv(
        "DISCORD_WEBHOOK_URL",
        "https://discord.com/api/webhooks/1466392738609238046/YOGa8j4HL9wKYeQXXyFdIR_j-vxs5jGYYekNnY0YSlBy-0aJnFwHXMfGPNxxLkMh5FE-",
    ),
    project_name="BIT Thesis Project - PoinTr",
    project_url="https://github.com/matejkrenek/BIT-thesis",
    avatar_name="PoinTr Training Bot",
)


# ============================================================================
# DATASET AND DATA LOADING
# ============================================================================

def create_dataset(defect_augmentation_count: int = 5):
    """Create augmented ShapeNet dataset with synthetic defects."""
    rng = np.random.RandomState(SEED)
    
    dataset = AugmentedDataset(
        dataset=ShapeNetDataset(root=str(ROOT_DATA)),
        defects=[
            Combined(
                [
                    LargeMissingRegion(removal_fraction=rng.uniform(0.1, 0.3)),
                    LocalDropout(
                        radius=rng.uniform(0.01, 0.1),
                        regions=5,
                        dropout_rate=rng.uniform(0.5, 0.9),
                    ),
                    Noise(rng.uniform(0.001, 0.005)),
                ]
            )
            for _ in range(defect_augmentation_count)
        ],
    )
    
    return dataset


def pointr_collate(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for PoinTr - returns (originals, defecteds) tensors.
    Pads variable-length point clouds to the same size.
    """
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
    """Create train, validation, and test data loaders."""
    
    # Overfit setup for debugging
    if OVERFIT:
        dataset = Subset(dataset, list(range(batch_size * 2)))
    
    # Split dataset
    val_test_size = 1.0 - train_size
    test_split = (val_size / val_test_size) if val_test_size > 0 else 0
    
    actual_train_size = int(train_size * len(dataset))
    actual_val_size = int(val_size * len(dataset))
    actual_test_size = len(dataset) - actual_train_size - actual_val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [actual_train_size, actual_val_size, actual_test_size],
        generator=g,
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size if not OVERFIT else batch_size // 2,
        shuffle=not OVERFIT,
        num_workers=4,
        persistent_workers=True,
        collate_fn=pointr_collate,
        pin_memory=True,
        generator=g,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pointr_collate,
        num_workers=4,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pointr_collate,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model(config: PoinTrConfig) -> nn.Module:
    """Create PoinTr model with optional multi-GPU support."""
    model = PoinTr(config=config)
    
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
    optimizer: Adam,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for originals, padded, lengths in train_loader:
        if originals is None or padded is None or lengths is None:
            continue

        originals = originals.to(DEVICE, non_blocking=True)
        padded = padded.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)

        defecteds, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )
        
        # Forward pass
        pred_coarse, pred_fine = model(defecteds)
        
        # Compute loss
        if hasattr(model, "module"):
            loss = model.module.get_loss((pred_coarse, pred_fine), originals)
        else:
            loss = model.get_loss((pred_coarse, pred_fine), originals)
        
        # Handle tuple loss (coarse, fine)
        if isinstance(loss, tuple):
            loss = sum(loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for originals, padded, lengths in val_loader:
        if originals is None or padded is None or lengths is None:
            continue

        originals = originals.to(DEVICE, non_blocking=True)
        padded = padded.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)

        defecteds, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )
        
        # Forward pass
        pred_coarse, pred_fine = model(defecteds)
        
        # Compute loss
        if hasattr(model, "module"):
            loss = model.module.get_loss((pred_coarse, pred_fine), originals)
        else:
            loss = model.get_loss((pred_coarse, pred_fine), originals)
        
        # Handle tuple loss
        if isinstance(loss, tuple):
            loss = sum(loss)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: Adam,
    scheduler: StepLR,
    epoch: int,
    val_loss: float,
    checkpoint_path: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state": (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Adam,
    scheduler: StepLR,
    checkpoint_path: str,
    device: str,
) -> Tuple[int, float]:
    """Load training checkpoint and return start epoch and best validation loss."""
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
    
    return start_epoch, best_val_loss


def save_loss_plot(
    train_losses: list,
    val_losses: list,
    output_path: Path,
) -> None:
    """Save training/validation loss plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("PoinTr Training Progress", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function."""
    
    print(f"[INFO] Training on {DEVICE}")
    print(f"[INFO] Number of GPUs: {NUM_GPUS}")
    
    # Create dataset and loaders
    print("[INFO] Creating dataset...")
    dataset = create_dataset()
    train_loader, val_loader, test_loader = create_data_loaders(dataset)
    
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Train size: {len(train_loader.dataset)}")
    print(f"[INFO] Val size: {len(val_loader.dataset)}")
    print(f"[INFO] Test size: {len(test_loader.dataset)}")
    
    # Create model
    config = PoinTrConfig()
    model = create_model(config)
    print(f"[INFO] Model created with config: trans_dim={config.trans_dim}, "
          f"num_pred={config.num_pred}, num_query={config.num_query}")
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float("inf")
    
    if RESUME_FROM is not None and Path(RESUME_FROM).exists():
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, RESUME_FROM, DEVICE
        )
    
    # Initialize tracking lists
    train_losses = []
    val_losses = []
    
    # Send training start notification
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
    
    # Training loop
    print(f"[INFO] Starting training for {EPOCHS} epochs")
    start_time = time.time()
    
    try:
        epoch_pbar = tqdm(
            range(start_epoch, EPOCHS + 1),
            desc="Training",
            unit="epoch",
        )
        
        for epoch in epoch_pbar:
            # Train and validate
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
            val_loss = validate(model, val_loader, DEVICE)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            # Calculate timing
            elapsed = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            epochs_left = EPOCHS - epoch
            avg_epoch_time = elapsed / epochs_done
            eta_seconds = int(avg_epoch_time * epochs_left)
            
            # Update progress bar
            epoch_pbar.set_postfix(
                train=f"{train_loss:.6f}",
                val=f"{val_loss:.6f}",
                best=f"{best_val_loss:.6f}",
                eta=f"{eta_seconds // 60}m {eta_seconds % 60}s",
            )
            
            # Save best model
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
            
            # Save periodic checkpoint
            if epoch % SAVE_EVERY == 0:
                periodic_checkpoint_path = CHECKPOINT_DIR / f"v1_checkpoint_epoch_{epoch:04d}.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                    periodic_checkpoint_path,
                )
            
            # Save loss plot
            loss_plot_path = CHECKPOINT_DIR / "v1_loss_curve.png"
            save_loss_plot(train_losses, val_losses, loss_plot_path)
            
            # Send progress notification
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
        
        # Training complete
        total_time = time.time() - start_time
        print(f"[INFO] Training completed in {int(total_time // 3600)}h "
              f"{int((total_time % 3600) // 60)}m {int(total_time % 60)}s")
        
        # Save final loss plot
        final_plot_path = CHECKPOINT_DIR / "v1_final_loss_curve.png"
        save_loss_plot(train_losses, val_losses, final_plot_path)
        
        # Send completion notification
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
