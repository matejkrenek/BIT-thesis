from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import os


class BaseDownloader(ABC):
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
