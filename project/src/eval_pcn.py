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
from models import PCN
import random as rnd
import matplotlib.pyplot as plt
from visualize.utils import plot_pointcloud_to_image
from metrics import chamfer_distance_metric, fscore_metric, hausdorff_distance_metric

# safe_gpu.claim_gpus(1)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_GPUS = torch.cuda.device_count()
print(f"[INFO] Available GPUs: {NUM_GPUS}")

BATCH_SIZE = 128
ROOT_DIR = os.getenv("ROOT_DIR", "")
ROOT_DATA = ROOT_DIR + "/data/ShapeNetV2"
CHECKPOINT_DIR = ROOT_DIR + "/checkpoints"
CHECKPOINT = CHECKPOINT_DIR + "/pcn_v69_best.pt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
THRESHOLD = 0.005

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


train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size], generator=g
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    collate_fn=pcn_collate,
    pin_memory=True,
    generator=g,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pcn_collate,
    num_workers=4,
    generator=g,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pcn_collate,
    num_workers=4,
    generator=g,
)

model = PCN(num_dense=16384, latent_dim=1024, grid_size=4)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

results = []

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

        cd = chamfer_distance_metric(pred, originals).item()
        hd = hausdorff_distance_metric(pred, originals).item()
        f1 = fscore_metric(pred, originals, THRESHOLD).item()

        plot_pointcloud_to_image(defected[0], "sample_defected_" + str(index) + ".png")
        plot_pointcloud_to_image(pred[0], "sample_predicted_" + str(index) + ".png")
        plot_pointcloud_to_image(originals[0], "sample_original_" + str(index) + ".png")

        results.append([cd, hd, f1])
        break

import pandas as pd

# -----------------------
# Save results
# -----------------------

df = pd.DataFrame(results, columns=["Chamfer", "Hausdorff", "F-score"])

df.to_csv("evaluation_results.csv", index=False)

print("\n===== SUMMARY =====")
print(df.describe())

# -----------------------
# Visualizations
# -----------------------

plt.figure(figsize=(8, 6))
plt.hist(df["Chamfer"], bins=30)
plt.title("Chamfer Distance Distribution")
plt.savefig("cd_histogram.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.boxplot(
    [
        df["Chamfer"],
        df["Hausdorff"],
        df["F-score"],
    ],
    tick_labels=["CD", "Hausdorff", "F-score"],
)
plt.title("Metric Distribution")
plt.savefig("metrics_boxplot.png")
plt.close()

print("[INFO] Evaluation completed.")
