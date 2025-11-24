from .base import Defect
import numpy as np


class Noise(Defect):
    name: str = "noise"

    def __init__(self, sigma: float = 0.005):
        self.sigma = sigma

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        noisy = points + np.random.normal(0, self.sigma, size=points.shape)
        return noisy, {"noise_sigma": self.sigma}
