from .base import BaseDownloader
import kagglehub
from logger import logger
import os
import shutil


class KaggleDownloader(BaseDownloader):
    """
    Downloader for datasets from Kaggle.
    Args:
        local_dir (str): Local directory to save the dataset.
        remote_dir (str): Kaggle dataset identifier in the format "owner/dataset-name".
        remote_file (Optional[str]): Specific file to download from the remote directory (not used in this implementation).
        token (Optional[str]): Authentication token for accessing private repositories (not used in this implementation).
        force (bool): Whether to force re-download if the dataset already exists locally.
    """

    def download(self) -> bool:
        """Download dataset from Kaggle if not already present locally."""
        # Check if dataset already exists locally
        if (
            os.path.exists(self.local_dir)
            and os.listdir(self.local_dir)
            and not self.force
        ):
            logger.info("[KaggleDownloader] Dataset already exists, skipping download.")
            return True

        # Download dataset using kagglehub
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

        # Move first subdirectory contents to local_dir
        shutil.copytree(first_subdir_path, self.local_dir)

        logger.info(
            f"[KaggleDownloader] Dataset moved to local directory: {self.local_dir}"
        )
        return True
