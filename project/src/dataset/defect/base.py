"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: base.py
Responsibility: Abstract base class for all synthetic defect transformations applied to point clouds.
"""

from abc import ABC, abstractmethod
import numpy as np


class Defect(ABC):
    """
    Abstract base class for all synthetic defect transformations applied to point clouds.

    Attributes:
        name (str): Name of the defect type.

    Methods:
        apply(points): Applies the defect to a point cloud.
    """

    name: str = "base_defect"

    @abstractmethod
    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply the defect transformation to a point cloud.

        Args:
            points (np.ndarray): Input point cloud of shape (N, 3).

        Returns:
            tuple: (defected_points, metadata_dict)
        """
        pass
