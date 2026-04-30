"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: scale.py
Responsibility: Synthetic defect class that scales point clouds by a given factor.
"""

from .base import Defect
import numpy as np


class Scale(Defect):
    """
    Synthetic defect that scales the point cloud by a given factor.

    Args:
        factor (float): Scaling factor to apply to all points.
    """

    name: str = "scale"

    def __init__(self, factor: float = 1.0):
        self.factor = float(factor)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        scaled = points * self.factor
        return scaled, {"factor": self.factor}
