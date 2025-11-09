import numpy as np
from .base import DefectInjector


class CombinedDefectInjector(DefectInjector):
    def __init__(self, defects: list[DefectInjector]):
        self.defects = defects

    def apply(self, points: np.ndarray) -> np.ndarray:
        corrupted = points
        for defect in self.defects:
            corrupted = defect.apply(corrupted)
        return corrupted
