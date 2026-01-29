from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import LargeMissingRegion, LocalDropout, Rotate, Noise
import open3d as o3d
import numpy as np
import os
from dotenv import load_dotenv
import torch
from torch.utils.data import Subset, DataLoader, random_split
import time
from pytorch3d.ops import sample_farthest_points
import safe_gpu
from tqdm import tqdm
from models import PCN
from notifications import DiscordNotifier

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_GPUS = torch.cuda.device_count()
print(f"[INFO] Available GPUs: {NUM_GPUS}")

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"
CHECKPOINT_DIR = DATA_FOLDER_PATH + "/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 100
SAVE_EVERY = 10  # checkpoint interval
RESUME_FROM = None  # e.g. "checkpoints/pcn_v2_epoch_50.pt"
OVERFIT = False  # True = overfit test

notifier = DiscordNotifier(
    webhook_url="https://discord.com/api/webhooks/1466392738609238046/YOGa8j4HL9wKYeQXXyFdIR_j-vxs5jGYYekNnY0YSlBy-0aJnFwHXMfGPNxxLkMh5FE-",
    project_name="BIT Thesis Project",
    project_url="https://github.com/matejkrenek/BIT-thesis",
    avatar_name="PCN Training Bot",
)

dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root=ROOT_DATA),
    defects=[
        LargeMissingRegion(removal_fraction=0.1),
        LargeMissingRegion(removal_fraction=0.15),
        LargeMissingRegion(removal_fraction=0.05),
        LargeMissingRegion(removal_fraction=0.25),
        LocalDropout(radius=0.05, regions=5, dropout_rate=0.8),
        Noise(0.002),
        Rotate(90, 90, 90),
    ],
)
train_losses = []
val_losses = []


def pcn_collate(batch):
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


# Overfit setup (1–2 samples only)
if OVERFIT:
    dataset = Subset(dataset, list(range(BATCH_SIZE)))
    BATCH_SIZE = int(BATCH_SIZE / 2)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=not OVERFIT,
    num_workers=4,
    persistent_workers=True,
    collate_fn=pcn_collate,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pcn_collate,
    num_workers=4,
)

model = PCN(num_dense=16384, latent_dim=1024, grid_size=4)

if DEVICE == "cuda" and NUM_GPUS > 1:
    print("[INFO] Using DataParallel")
    model = torch.nn.DataParallel(model)

model = model.to(DEVICE)

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

    state = checkpoint["model_state"]
    if hasattr(model, "module"):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val = checkpoint["val_loss"]


def train_epoch():
    model.train()
    total_loss = 0.0

    for originals, padded, lengths in train_loader:
        if originals is None or padded is None or lengths is None:
            continue

        originals = originals.to(DEVICE, non_blocking=True)
        padded = padded.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)

        defected, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )

        pred = model(defected)

        if hasattr(model, "module"):
            loss = model.module.compute_loss(pred, originals)
        else:
            loss = model.compute_loss(pred, originals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def val_epoch():
    model.eval()
    total_loss = 0.0

    for originals, padded, lengths in val_loader:
        if originals is None or padded is None or lengths is None:
            continue

        originals = originals.to(DEVICE, non_blocking=True)
        padded = padded.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)

        defected, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )

        pred = model(defected)

        if hasattr(model, "module"):
            loss = model.module.compute_loss(pred, originals)
        else:
            loss = model.compute_loss(pred, originals)

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

import matplotlib.pyplot as plt


def save_loss_plot(train_losses, val_losses, path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Chamfer Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


current_epoch = start_epoch - 1

notifier.send_training_start(
    total_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    train_size=train_size,
    val_size=val_size,
    training_on=DEVICE,
    number_of_gpus=NUM_GPUS,
    learning_rate=LR,
)

try:
    for epoch in epoch_bar:
        current_epoch = epoch
        train_loss = train_epoch()
        val_loss = val_epoch()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_loss_plot(train_losses, val_losses, "loss_curve.png")

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

        notifier.send_training_progress(
            epoch=epoch,
            total_epochs=EPOCHS,
            current_loss=val_loss,
            best_loss=best_val,
            learning_rate=scheduler.get_last_lr()[0],
            batch_size=BATCH_SIZE,
            elapsed_time=f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
            estimated_finish_time=time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta_seconds)
            ),
            loss_curve_path="./loss_curve.png",
        )

        # Save BEST model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    ),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(CHECKPOINT_DIR, "pcn_v2_best.pt"),
            )

    save_loss_plot(train_losses, val_losses, "final_loss_curve.png")

    print("[INFO] Training finished.")

    notifier.send_training_completion(
        total_epochs=EPOCHS,
        final_loss=val_losses[-1],
        best_loss=best_val,
        training_time=f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
        final_loss_curve_path="./final_loss_curve.png",
        best_model_path="./checkpoints/pcn_v2_best.pth",
    )
except Exception as e:
    print(f"[ERROR] Training interrupted: {e}")
    notifier.send_training_error(error_message=str(e), epoch=current_epoch)
