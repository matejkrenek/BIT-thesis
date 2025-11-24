from .base import Defect
import numpy as np


class Translate(Defect):
    name: str = "translate"

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.offset = np.array([x, y, z], dtype=float)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        translated = points + self.offset

        return translated, {
            "x": self.offset[0],
            "y": self.offset[1],
            "z": self.offset[2],
        }
