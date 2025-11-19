from .base import BaseDownloader
from huggingface_hub import hf_hub_download, list_repo_files
from logger import logger


class HuggingFaceDownloader(BaseDownloader):
    """Downloader for datasets hosted on Hugging Face Hub."""

    def __init__(self, repo_id: str, local_dir: str, token: str):
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.token = token

    def get_files(self):
        """Retrieve the list of files in the repository."""
        return list_repo_files(self.repo_id, token=self.token, repo_type="dataset")

    def download(self):
        """Download the dataset from Hugging Face Hub."""
        files = self.get_files()
        for file_name in files:
            logger.info(f"Downloading {file_name} from {self.repo_id}...")
            hf_hub_download(
                repo_id=self.repo_id,
                filename=file_name,
                local_dir=self.local_dir,
                token=self.token,
                repo_type="dataset",
            )
            logger.info(f"Downloaded {file_name} to {self.local_dir}.")
