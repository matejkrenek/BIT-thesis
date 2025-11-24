from torch_geometric.datasets import ShapeNet
from logger import logger
import kagglehub
import shutil
import os


class ShapeNetDataset(ShapeNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info(f"[ShapeNetDataset] Dataset initialized with {len(self)} samples.")

    def download(self) -> None:
        if os.path.exists(self.raw_dir) and os.listdir(self.raw_dir):
            logger.info("[ShapeNetDataset] Dataset already exists, skipping download.")
            return

        logger.info("[ShapeNetDataset] Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("mitkir/shapenet")
        logger.info(f"[ShapeNetDataset] Dataset downloaded to: {path}")

        # Moving downloaded dataset to the expected raw_dir
        if os.path.exists(self.raw_dir):
            shutil.rmtree(self.raw_dir)

        shutil.copytree(
            os.path.join(
                path, "shapenetcore_partanno_segmentation_benchmark_v0_normal"
            ),
            self.raw_dir,
        )
        logger.info(f"[ShapeNetDataset] Dataset moved to raw directory: {self.raw_dir}")
