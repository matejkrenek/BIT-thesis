from torch_geometric.datasets import ShapeNet
from logger import logger
from dataset.config import DatasetConfig


class ShapeNetDataset(ShapeNet):
    def __init__(self, config: DatasetConfig):
        self.config = config
        super().__init__(root=str(config.local_dir), categories=config.categories)

        logger.info(f"ShapeNetDataset initialized with {len(self)} models.")

    def download(self):

        logger.info("[ShapeNetDataset] Skipping download() – using local dataset only.")
