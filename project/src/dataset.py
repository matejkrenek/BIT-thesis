from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import (
    LargeMissingRegion,
    LocalDropout,
    Noise,
    FloatingCluster,
    AnisotropicStretchNoise,
    OutlierPoints,
)
import open3d as o3d
import numpy as np
import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
import time
import polyscope as ps

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"

dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root=ROOT_DATA),
    defects=[
        LargeMissingRegion(removal_fraction=0.25),
        LargeMissingRegion(removal_fraction=0.25),
        LocalDropout(radius=0.05, regions=3, dropout_rate=1),
        Noise(sigma=0.005),
        OutlierPoints(num_points=120, scale_factor=2.0, mode="uniform"),
    ],
)

ps.init()

original, defected = dataset[0]

ps.register_point_cloud(
    "original",
    original,
    radius=0.0025,
    color=(0.0, 1.0, 0.0),
)

ps.register_point_cloud(
    "defected1",
    dataset[0][1],
    radius=0.0025,
    color=(1.0, 0.0, 0.0),
)

ps.register_point_cloud(
    "defected2",
    dataset[1][1],
    radius=0.0025,
    color=(1.0, 0.0, 0.0),
)

ps.register_point_cloud(
    "defected3",
    dataset[2][1],
    radius=0.0025,
    color=(1.0, 0.0, 0.0),
)

ps.register_point_cloud(
    "defected4",
    dataset[7][1],
    radius=0.0025,
    color=(1.0, 0.0, 0.0),
)


ps.show()


exit()

from pytorch3d.ops import sample_farthest_points


def pcn_collate(batch):
    originals, defecteds = zip(*batch)

    originals = torch.stack(originals, dim=0)

    lengths = torch.tensor([pc.shape[0] for pc in defecteds], dtype=torch.long)
    max_n = lengths.max().item()

    padded = torch.zeros(len(defecteds), max_n, 3)
    for i, pc in enumerate(defecteds):
        padded[i, : pc.shape[0]] = pc

    return originals, padded, lengths


dataset_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=pcn_collate,
    persistent_workers=True,
    pin_memory=True,
)

import polyscope as ps

for originals, padded, lengths in dataset_loader:
    originals = originals.to(DEVICE, non_blocking=True)
    padded = padded.to(DEVICE, non_blocking=True)
    lengths = lengths.to(DEVICE)

    defected, _ = sample_farthest_points(
        padded,
        K=originals.shape[1],
        lengths=lengths,
    )

    print(defected[0])

    break
