from abc import ABC, abstractmethod


class BaseDownloader(ABC):
    """Abstract base class for dataset downloaders."""

    @abstractmethod
    def download(self, force: bool = False) -> bool:
        pass
