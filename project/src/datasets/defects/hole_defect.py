from .base import DefectInjector
import numpy as np


class HoleDefect(DefectInjector):
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def apply(self, points: np.ndarray) -> np.ndarray:
        pass
