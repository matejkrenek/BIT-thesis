from .base import BaseDownloader
from huggingface_hub import list_repo_files, hf_hub_download


class HuggingFaceDownloader(BaseDownloader):

    def _list_files(self):
        return list_repo_files(
            repo_id=self.config.repo_id, token=self.config.token, repo_type="dataset"
        )

    def download(self) -> bool:
        """
        Download dataset from HuggingFace Hub.

        Returns:
            bool: True if download was successful, False otherwise.
        """
        files = self._list_files()

        # Check if the specified file exists in the repository
        if self.config.repo_file and self.config.repo_file not in files:
            raise ValueError(
                f"File {self.config.repo_file} not found in repository {self.config.repo_id}."
            )

        # If a specific file is requested, filter the list
        if self.config.repo_file:
            files = [self.config.repo_file]

        # Download each file
        for file in files:
            hf_hub_download(
                repo_id=self.config.repo_id,
                filename=file,
                token=self.config.token,
                repo_type="dataset",
                local_dir=str(self.config.local_dir),
            )

        return True
