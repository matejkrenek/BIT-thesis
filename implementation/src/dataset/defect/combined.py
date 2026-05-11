"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: combined.py
Responsibility: Synthetic defect class that applies multiple defect types in sequence to a point cloud.
"""

from .base import Defect
import numpy as np


class Combined(Defect):
    """
    Synthetic defect that applies multiple defect types in sequence to a point cloud.

    Args:
        defects (list[Defect]): List of defect instances to apply in order.
    """

    name: str = "combined"

    def __init__(
        self,
        defects: list[Defect],
    ):
        self.defects = defects

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        metadata = {}
        for defect in self.defects:
            points, defect_metadata = defect.apply(points)
            metadata[defect.name] = defect_metadata
        return points, metadata
