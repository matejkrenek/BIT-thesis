import subprocess
from pathlib import Path
from .base import BaseDownloader
from logger import logger
import shutil


class GitDownloader(BaseDownloader):
    """
    Generic git-based downloader.

    Assumes repo_url already contains authentication token (if needed).
    """

    def __init__(self, repo_url: str, local_dir: str):
        self.repo_url = repo_url
        self.local_dir = Path(local_dir)

    def _git_available(self) -> bool:
        try:
            subprocess.run(["git", "--version"], stdout=subprocess.PIPE)
            subprocess.run(["git", "lfs", "install"], stdout=subprocess.PIPE)
            subprocess.run(["git", "lfs", "pull"], stdout=subprocess.PIPE)
            return True
        except FileNotFoundError:
            return False

    def _git_clone(self, force: bool = False) -> bool:
        if self.local_dir.exists() and any(self.local_dir.iterdir()) and not force:
            logger.info(
                f"Directory already contains data: {self.local_dir}. Skipping git clone."
            )
            return True

        logger.info(f"Cloning repository: {self.repo_url}")

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.local_dir)],
                check=True,
            )
            logger.success("Git clone completed.")
            return True
        except subprocess.CalledProcessError:
            logger.error("Git clone failed.")
            return False

    def download(self, force: bool = False) -> bool:
        if not self._git_available():
            logger.error("Git is not installed. Cannot download dataset.")
            return False

        # Create local directory if it doesn't exist
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Remove directory if force is True and it's not empty
        if force and self.local_dir.exists() and any(self.local_dir.iterdir()):
            logger.info(
                f"Force flag set. Removing existing directory: {self.local_dir}"
            )
            shutil.rmtree(self.local_dir)
            self.local_dir.mkdir(parents=True, exist_ok=True)

        return self._git_clone(force=force)
