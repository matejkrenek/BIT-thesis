from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import (
    LargeMissingRegion,
    LocalDropout,
    Noise,
    FloatingCluster,
    AnisotropicStretchNoise,
    OutlierPoints,
    Combined,
    Rotate,
)
import open3d as o3d
import numpy as np
import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, random_split
import time
import polyscope as ps
from visualize.viewer import SampleViewer
import random as rnd

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"

SEED = 42
g = torch.Generator()
g.manual_seed(SEED)
rng = np.random.RandomState(SEED)

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
                Rotate(0, 0, rng.uniform(0, 360)),
            ]
        )
        for _ in range(10)
    ],
    detailed=True,
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    generator=g,
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    generator=g,
)


viewer = SampleViewer(dataset=train_loader.dataset)
viewer.show()

# viewer = SampleViewer(dataset=dataset)
# viewer.show()
