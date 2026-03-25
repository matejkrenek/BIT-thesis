import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from dataset import ShapeNetDataset
from dataset.wrapper import (
    DenseWrapperDataset,
    SplitWrapperDataset,
    NormalizeWrapperDataset,
    AugmentWrapperDataset,
)
from dataset.defect import LargeMissingRegion, Rotate, Noise, LocalDropout, Combined
import polyscope as ps

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"

base_dataset = ShapeNetDataset(
    root=ROOT_DATA,
)

dense_dataset = DenseWrapperDataset(
    dataset=base_dataset,
    root=DATA_FOLDER_PATH + "/data/ShapeNetV2_dense",
    num_points=100_000,
)

normalized_base_dataset = NormalizeWrapperDataset(base_dataset)
normalized_dense_dataset = NormalizeWrapperDataset(dense_dataset)

defects = [
    Combined(
        [
            LargeMissingRegion(removal_fraction=0.3),
            LocalDropout(
                radius=0.1,
                regions=5,
                dropout_rate=0.5,
            ),
        ]
    )
    for _ in range(5)
]


augmented_base_dataset = AugmentWrapperDataset(normalized_base_dataset, defects)
augmented_dense_dataset = AugmentWrapperDataset(normalized_dense_dataset, defects)


ps.init()

ps.register_point_cloud(
    "original",
    augmented_dense_dataset[0].original_pos,
    radius=0.001,
    color=(0.0, 1.0, 0.0),
    point_render_mode="quad",
)
ps.register_point_cloud(
    "defected",
    augmented_dense_dataset[0].defected_pos,
    radius=0.001,
    color=(1.0, 0.0, 0.0),
    point_render_mode="quad",
)

ps.show()
