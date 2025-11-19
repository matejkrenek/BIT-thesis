from abc import ABC, abstractmethod
from ..config import DatasetDownloadConfig


class BaseDownloader(ABC):
    def __init__(self, config: DatasetDownloadConfig):
        self.config = config

    @abstractmethod
    def download(self) -> bool:
        pass
