from dataset import ShapeNetDataset, AugmentedDataset, ModelNetDataset, CO3DDataset, PhotogrammetricDataset
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
from dataset.downloader import ZipUrlDownloader
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
from pathlib import Path
import shutil
from visualize.utils import plot_pointcloud_to_image, plot_dense_pointcloud_to_image

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"

SEED = 42
g = torch.Generator()
g.manual_seed(SEED)
rng = np.random.RandomState(SEED)
np.random.seed(SEED)

dataset = PhotogrammetricDataset(
    dataset=CO3DDataset(root=DATA_FOLDER_PATH + "/data/CO3D", categories=["cup", "bench", "car"], samples_per_category=10), 
    frames_per_sample=[5, 10, 20, 25], 
    frames_strategy="uniform", 
)

sample = dataset[0]
print(sample[0])
plot_dense_pointcloud_to_image(sample[0], output_path=f"./sample_0_unmasked.png")
plot_dense_pointcloud_to_image(sample[1], output_path=f"./sample_0_masked.png")
exit(0)
for i in range(len(dataset)):
    try:
        sample = dataset[i]
        plot_dense_pointcloud_to_image(sample[0], output_path=f"./sample_{i}_unmasked.png")
        plot_dense_pointcloud_to_image(sample[1], output_path=f"./sample_{i}_masked.png")

        break
    except Exception as e:
        continue

exit(0)

pl = o3d.io.read_point_cloud("./pointcloud.ply")

ps.init()

ps.register_point_cloud("original", np.asarray(pl.points))

ps.show()

exit(0)

dataset = AugmentedDataset(
    dataset=ModelNetDataset(root=DATA_FOLDER_PATH + "/data/ModelNet40"),
    defects=[
        Combined(
            [
                LargeMissingRegion(removal_fraction=rng.uniform(0.1, 0.3)),
                LocalDropout(
                    radius=rng.uniform(0.01, 0.1),
                    regions=5,
                    dropout_rate=rng.uniform(0.5, 0.9),
                ),
                Noise(rng.uniform(0.005, 0.01)),
            ]
        )
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
