import os
from dotenv import load_dotenv
from dataset.utils import (
    create_basic_reconstruction_dataset,
    create_advanced_reconstruction_dataset,
)

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")

basic_dataset = create_basic_reconstruction_dataset(
    root=DATA_FOLDER_PATH + "/data/ShapeNetV2",
    normalize=True,
    visualize=True,
)
