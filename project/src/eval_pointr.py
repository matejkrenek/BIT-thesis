from dataset import AugmentedDataset, ShapeNetDataset
from dataset.defect import LargeMissingRegion, LocalDropout, Noise, Combined
import open3d as o3d
import numpy as np
import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, random_split
from pytorch3d.ops import sample_farthest_points
import safe_gpu
from tqdm import tqdm
from models import PCN, PoinTr
import random as rnd
import matplotlib.pyplot as plt
from visualize.utils import plot_pointcloud_to_image
from typing import Optional, Tuple

# safe_gpu.claim_gpus(1)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_GPUS = torch.cuda.device_count()
print(f"[INFO] Available GPUs: {NUM_GPUS}")

BATCH_SIZE = 128
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"
CHECKPOINT_DIR = DATA_FOLDER_PATH + "/checkpoints/pointr"
CHECKPOINT = CHECKPOINT_DIR + "/v1_best.pt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
THRESHOLD = 0.05

SEED = 42
g = torch.Generator()
g.manual_seed(SEED)
rng = np.random.RandomState(SEED)
np.random.seed(SEED)

dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root=ROOT_DATA),
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
        for _ in range(5)
    ],
)

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

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=g)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    collate_fn=pointr_collate,
    pin_memory=True,
    generator=g,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pointr_collate,
    num_workers=4,
    generator=g,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pointr_collate,
    num_workers=4,
    generator=g,
)

# PoinTr specific configuration
class PoinTrConfig:
    trans_dim = 384
    knn_layer = 0
    num_pred = 16384
    num_query = 128

model = PoinTr(config=PoinTrConfig())
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

from pytorch3d.ops import sample_farthest_points, knn_points


def compute_hd95(pred, gt):
    knn = knn_points(pred, gt, K=1)
    dists = knn.dists.squeeze(-1)
    return torch.quantile(dists, 0.95).item()

def compute_fscore(pred, gt, threshold):
    knn1 = knn_points(pred, gt, K=1)
    knn2 = knn_points(gt, pred, K=1)

    dist1 = knn1.dists.squeeze(-1)
    dist2 = knn2.dists.squeeze(-1)

    precision = (dist1 < threshold).float().mean()
    recall = (dist2 < threshold).float().mean()

    return (2 * precision * recall / (precision + recall + 1e-8)).item()

def compute_nn_variance(pc):
    knn = knn_points(pc, pc, K=2)
    dists = knn.dists[:, :, 1]
    return torch.var(dists).item()

results = []

from pytorch3d.loss import chamfer_distance

with torch.no_grad():
    index = 0

    for originals, padded, lengths in tqdm(test_loader):

        originals = originals.to(DEVICE)
        padded = padded.to(DEVICE)
        lengths = lengths.to(DEVICE)

        defected, _ = sample_farthest_points(
            padded,
            K=originals.shape[1],
            lengths=lengths,
        )

        coarse, pred = model(defected)

        pred, _ = sample_farthest_points(
            pred,
            K=originals.shape[1],
        )

        cd, _ = chamfer_distance(pred, originals)
        hd95 = compute_hd95(pred, originals)
        f1 = compute_fscore(pred, originals, THRESHOLD)
        nn_var = compute_nn_variance(pred)

        plot_pointcloud_to_image(defected[0], "sample_defected_" + str(index) +  ".png")
        plot_pointcloud_to_image(pred[0], "sample_predicted_" + str(index) + ".png")
        plot_pointcloud_to_image(originals[0], "sample_original_" + str(index) + ".png")
        
        results.append([cd.item(), hd95, f1, nn_var])
        break

import pandas as pd

# -----------------------
# Save results
# -----------------------

df = pd.DataFrame(
    results,
    columns=["Chamfer", "HD95", "F1@1%", "NN_Variance"]
)

df.to_csv("evaluation_results.csv", index=False)

print("\n===== SUMMARY =====")
print(df.describe())

# -----------------------
# Visualizations
# -----------------------

plt.figure(figsize=(8,6))
plt.hist(df["Chamfer"], bins=30)
plt.title("Chamfer Distance Distribution")
plt.savefig("cd_histogram.png")
plt.close()

plt.figure(figsize=(8,6))
plt.boxplot([
    df["Chamfer"],
    df["HD95"],
    df["F1@1%"],
    df["NN_Variance"]
], tick_labels=["CD", "HD95", "F1", "NN_Var"])
plt.title("Metric Distribution")
plt.savefig("metrics_boxplot.png")
plt.close()

print("[INFO] Evaluation completed.")