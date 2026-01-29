import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import LargeMissingRegion, Rotate, Noise, LocalDropout, Combined
from models import PCN
from pytorch3d.ops import sample_farthest_points
from pytorch3d.loss import chamfer_distance
import random as rnd

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"
CHECKPOINT_PATH = DATA_FOLDER_PATH + "/checkpoints/pcn_v2_best.pt"

BATCH_SIZE = 16
NUM_WORKERS = 4

# -----------------------
# Dataset
# -----------------------

test_dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root=ROOT_DATA),
    defects=[
        Combined(
            [
                LargeMissingRegion(removal_fraction=rnd.uniform(0.1, 0.3)),
                LocalDropout(
                    radius=rnd.uniform(0.01, 0.1),
                    regions=5,
                    dropout_rate=rnd.uniform(0.5, 0.9),
                ),
                Noise(rnd.uniform(0.001, 0.005)),
                Rotate(0, 0, rnd.uniform(0, 360)),
            ]
        )
        for _ in range(10)
    ],
)


def pcn_collate(batch):
    originals, defecteds = zip(*batch)

    originals = torch.stack(originals, dim=0)

    lengths = torch.tensor([pc.shape[0] for pc in defecteds], dtype=torch.long)
    max_n = lengths.max().item()

    padded = torch.zeros(len(defecteds), max_n, 3)
    for i, pc in enumerate(defecteds):
        padded[i, : pc.shape[0]] = pc

    return originals, padded, lengths


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=pcn_collate,
    pin_memory=True,
)

# -----------------------
# Model
# -----------------------

model = PCN(
    num_dense=16384,
    latent_dim=1024,
    grid_size=4,
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"[INFO] Loaded checkpoint: {CHECKPOINT_PATH}")
print(f"[INFO] Evaluating on {len(test_dataset)} samples")

# -----------------------
# Evaluation
# -----------------------


@torch.no_grad()
def evaluate():
    total_cd = 0.0

    for originals, padded, lengths in tqdm(test_loader, desc="Testing"):
        originals = originals.to(DEVICE, non_blocking=True)
        padded = padded.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)

        defected, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )

        coarse, fine = model(defected)

        cd_fine, _ = chamfer_distance(fine, originals)
        total_cd += cd_fine.item()

    mean_cd = total_cd / len(test_loader)
    return mean_cd


mean_cd = evaluate()

print(f"[RESULT] Mean Chamfer Distance (fine): {mean_cd:.6f}")
