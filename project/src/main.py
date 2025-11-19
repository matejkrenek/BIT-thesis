import dotenv
import os
from pathlib import Path
from dataset.config import DatasetConfig, DatasetDownloadConfig
from dataset.shapenet import ShapeNetDataset
from dataset.downloader.huggingface import HuggingFaceDownloader
import polyscope as ps
import numpy as np
import open3d as o3d

dotenv.load_dotenv()

dataset_config = DatasetConfig(
    name="ShapeNetCore",
    local_dir=Path("data/ShapeNetCore/raw"),
    categories=["Pistol"],
    download=DatasetDownloadConfig(
        repo_id="ShapeNet/ShapeNetCore",
        token=os.getenv("HUGGING_FACE_TOKEN", ""),
        local_dir=Path("data/ShapeNetCore/raw"),
    ),
)

dataset = ShapeNetDataset(config=dataset_config)
