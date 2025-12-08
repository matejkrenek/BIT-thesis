from abc import ABC, abstractmethod


class BaseViewer(ABC):
    """Abstract base class for all viewers."""

    initialized = False

    @abstractmethod
    def show(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass
