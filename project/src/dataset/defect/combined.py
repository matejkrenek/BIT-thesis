from .base import Defect
import numpy as np


class Combined(Defect):
    """
    Combines multiple defect types into a single defect application.
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
