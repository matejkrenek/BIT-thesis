from abc import ABC, abstractmethod
import numpy as np


class DefectInjector(ABC):
    @abstractmethod
    def apply(self, points: np.ndarray) -> np.ndarray:
        """Inject efect to input point cloud"""
        pass
