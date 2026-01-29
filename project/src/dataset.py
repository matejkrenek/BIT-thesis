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
from torch.utils.data import DataLoader
import time
import polyscope as ps
from visualize.viewer import SampleViewer
import random as rnd

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"

dataset = AugmentedDataset(
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
    detailed=True,
)

viewer = SampleViewer(dataset=dataset)
viewer.show()

viewer = SampleViewer(dataset=dataset)
viewer.show()
