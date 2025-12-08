from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path


class BaseDownloader(ABC):
    """
    Abstract base class for all dataset downloaders.
    Args:
        local_dir (str): Local directory to save the dataset.
        remote_dir (str): Remote directory or repository identifier.
        remote_file (Optional[str]): Specific file to download from the remote directory.
        token (Optional[str]): Authentication token for accessing private repositories.
        force (bool): Whether to force re-download if the dataset already exists locally.
    """

    def __init__(
        self,
        local_dir: str,
        remote_dir: str,
        remote_file: Optional[str] = None,
        token: Optional[str] = None,
        force: bool = False,
    ):
        self.local_dir = Path(local_dir)
        self.remote_dir = remote_dir
        self.remote_file = remote_file
        self.token = token
        self.force = force

        if not self.local_dir.exists():
            self.local_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> bool:
        pass
