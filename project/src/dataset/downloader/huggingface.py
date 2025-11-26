from .base import BaseDownloader
from huggingface_hub import list_repo_files, hf_hub_download
from logger import logger
import zipfile
import os
from pathlib import Path


class HuggingFaceDownloader(BaseDownloader):

    def _list_files(self):
        return list_repo_files(
            repo_id=self.remote_dir, token=self.token, repo_type="dataset"
        )

    def download(self) -> bool:
        if (
            os.path.exists(self.local_dir)
            and os.listdir(self.local_dir)
            and not self.force
        ):
            logger.info(
                "[HuggingFaceDownloader] Dataset already exists, skipping download."
            )
            return True

        files = self._list_files()

        # Check if the specified file exists in the repository
        if self.remote_file:
            # Ensure repo_file is a list
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
        # Download each file
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

        logger.info("[HuggingFaceDownloader] Removing cache...")
        Path(os.path.join(self.local_dir, ".cache")).unlink(missing_ok=True)

        logger.info(f"[HuggingFaceDownloader] Dataset downloaded to: {self.local_dir}")
        return True
