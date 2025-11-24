from abc import ABC, abstractmethod
import numpy as np


class Defect(ABC):
    name: str = "base_defect"

    @abstractmethod
    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        pass
