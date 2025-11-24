from .base import Defect
import numpy as np


class Scale(Defect):
    name: str = "scale"

    def __init__(self, factor: float = 1.0):
        self.factor = float(factor)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        scaled = points * self.factor

        return scaled, {"factor": self.factor}
