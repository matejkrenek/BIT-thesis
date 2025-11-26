from .base import BaseDownloader
import kagglehub
from logger import logger
import os
import shutil


class KaggleDownloader(BaseDownloader):
    def download(self) -> bool:
        if (
            os.path.exists(self.local_dir)
            and os.listdir(self.local_dir)
            and not self.force
        ):
            logger.info("[KaggleDownloader] Dataset already exists, skipping download.")
            return True

        logger.info("[KaggleDownloader] Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download(self.remote_dir)
        logger.info(f"[KaggleDownloader] Dataset downloaded to: {self.local_dir}")

        # Moving downloaded dataset to the expected local_dir
        if os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)

        # Find the first directory within the downloaded path
        first_subdir_path = path
        for item in os.listdir(path):
            full_item_path = os.path.join(path, item)
            if os.path.isdir(full_item_path):
                first_subdir_path = full_item_path
                break

        shutil.copytree(first_subdir_path, self.local_dir)

        logger.info(
            f"[KaggleDownloader] Dataset moved to local directory: {self.local_dir}"
        )
        return True
