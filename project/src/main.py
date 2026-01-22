from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import (
    LargeMissingRegion,
)
from visualize.viewer import SampleViewer
from dotenv import load_dotenv
import open3d as o3d
import numpy as np
from torch.utils.data import Subset, DataLoader, random_split
from models.pcn import PCNRepairNet
import torch
import os
from tqdm import tqdm
import time
import safe_gpu

safe_gpu.claim_gpus(1)

# Suppress Open3D log messages and load environment variables
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA =  DATA_FOLDER_PATH + "/data/ShapeNetV2"
CHECKPOINT_DIR = DATA_FOLDER_PATH + "/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 200
SAVE_EVERY = 10  # checkpoint interval
RESUME_FROM = None  # e.g. "checkpoints/pcn_epoch_50.pt"
OVERFIT = False  # True = overfit test

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

# Create an augmented dataset from shape net
dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root=ROOT_DATA),
    defects=[
        LargeMissingRegion(removal_fraction=0.1),
        # OutlierPoints(num_points=50, scale_factor=2.0, mode="gaussian"),
        # SurfaceFlattening(radius=0.1, plane_jitter=0.0002, max_regions=1),
    ],
)

# viewer = SampleViewer(dataset)
# viewer.show()

# Overfit setup (1–2 samples only)
if OVERFIT:
    dataset = Subset(dataset, [0, 1])

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds,
    batch_size=1 if OVERFIT else BATCH_SIZE,
    shuffle=not OVERFIT,
    num_workers=12,
    persistent_workers=True,
    collate_fn=safe_collate,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=1 if OVERFIT else BATCH_SIZE,
    shuffle=False,
    collate_fn=safe_collate,
    num_workers=12,
)

model = PCNRepairNet(
    feat_dim=1024,
    num_coarse=1024,
    grid_size=4,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,
    gamma=0.5,
)

start_epoch = 1
best_val = float("inf")

if RESUME_FROM is not None:
    print(f"[INFO] Resuming from checkpoint: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val = checkpoint["val_loss"]


def train_epoch():
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        if batch is None:
            continue
        original, defected = batch
        defected = defected.to(DEVICE, non_blocking=True)
        original = original.to(DEVICE, non_blocking=True)

        pred = model(defected)
        loss = model.compute_loss(pred, original)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def val_epoch():
    model.eval()
    total_loss = 0.0

    for batch in val_loader:
        if batch is None:
            continue
        original, defected = batch
        defected = defected.to(DEVICE)
        original = original.to(DEVICE)

        pred = model(defected)
        loss = model.compute_loss(pred, original)

        total_loss += loss.item()

    return total_loss / len(val_loader)


print(f"[INFO] Training on {DEVICE}")
print(f"[INFO] Dataset size: {len(dataset)}")

epoch_bar = tqdm(
    range(start_epoch, EPOCHS + 1),
    desc="Training",
    unit="epoch",
)

start_time = time.time()

for epoch in epoch_bar:
    train_loss = train_epoch()
    val_loss = val_epoch()


    scheduler.step()

    elapsed = time.time() - start_time
    epochs_done = epoch - start_epoch + 1
    epochs_left = EPOCHS - epoch

    avg_epoch_time = elapsed / epochs_done
    eta_seconds = int(avg_epoch_time * epochs_left)

    epoch_bar.set_postfix(
        train=f"{train_loss:.4f}",
        val=f"{val_loss:.4f}",
        eta=f"{eta_seconds//60}m {eta_seconds%60}s",
    )

    # Save BEST model
    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(CHECKPOINT_DIR, "pcn_best.pt"),
        )

    # Periodic checkpoint
    if epoch % SAVE_EVERY == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(CHECKPOINT_DIR, f"pcn_epoch_{epoch}.pt"),
        )

print("[INFO] Training finished.")
