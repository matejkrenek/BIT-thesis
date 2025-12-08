from .base import BaseDownloader
from huggingface_hub import list_repo_files, hf_hub_download
from logger import logger
import zipfile
import os


class HuggingFaceDownloader(BaseDownloader):
    """
    Downloader for datasets from HuggingFace Hub.
    Args:
        local_dir (str): Local directory to save the dataset.
        remote_dir (str): HuggingFace dataset repository identifier.
        remote_file (Optional[str]): Specific file or list of files to download from the remote directory.
        token (Optional[str]): Authentication token for accessing private repositories.
        force (bool): Whether to force re-download if the dataset already exists locally.
    """

    def _list_files(self):
        """List all files in the remote HuggingFace dataset repository."""
        return list_repo_files(
            repo_id=self.remote_dir, token=self.token, repo_type="dataset"
        )

    def download(self) -> bool:
        """Download dataset from HuggingFace Hub if not already present locally."""
        # Check if dataset already exists locally
        if (
            os.path.exists(self.local_dir)
            and os.listdir(self.local_dir)
            and not self.force
        ):
            logger.info(
                "[HuggingFaceDownloader] Dataset already exists, skipping download."
            )
            return True

        # List files in the remote repository
        files = self._list_files()

        # Check if the specified file to be downloaded exists
        if self.remote_file:
            repo_files = (
                self.remote_file
                if isinstance(self.remote_file, list)
                else [self.remote_file]
            )

            # Check if all specified files exist in the repository
            missing_files = [f for f in repo_files if f not in files]
            if missing_files:
                raise ValueError(
                    f"Files {missing_files} not found in repository {self.remote_dir}."
                )

        # If a specific file is requested, filter the list
        if self.remote_file:
            files = self.remote_file

        logger.info(f"[HuggingFaceDownloader] Downloading dataset from HuggingFace...")

        # Download each file by file
        for file in files:
            logger.info(f"[HuggingFaceDownloader] Downloading file: {file}")
            hf_hub_download(
                repo_id=self.remote_dir,
                filename=file,
                token=self.token,
                repo_type="dataset",
                local_dir=str(self.local_dir),
            )
            logger.info(f"[HuggingFaceDownloader] Downloaded file: {file}")

        logger.info("[HuggingFaceDownloader] Unzipping files...")

        # Unzip all zips in the local directory
        for file in self.local_dir.iterdir():
            if file.suffix == ".zip":
                logger.info(f"[HuggingFaceDownloader] Unzipping file: {file}")
                with zipfile.ZipFile(file) as zip_ref:
                    zip_ref.extractall(self.local_dir)
                logger.info(f"[HuggingFaceDownloader] Unzipped file: {file}")
                file.unlink(missing_ok=True)

        logger.info(f"[HuggingFaceDownloader] Dataset downloaded to: {self.local_dir}")
        return True
