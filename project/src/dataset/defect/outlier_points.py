from .base import Defect
import numpy as np


class OutlierPoints(Defect):
    """
    OutlierPoints adds random outlier points around the main object bounding box. Caused by environmental noise or sensor errors during MVS.
    """

    name: str = "outlier_points"

    def __init__(
        self,
        num_points: int = 50,
        scale_factor: float = 1.5,
        mode: str = "uniform",
    ):
        """
        Args:
            num_points (int): Number of outlier points to add.
            scale_factor (float): Factor to determine how far outliers are from the main bounding box.
            mode (str): Distribution mode for outlier placement. Currently only 'uniform' or 'gaussian' is supported.
        """
        self.num_points = int(num_points)
        self.scale_factor = float(scale_factor)
        self.mode = mode

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        # Nothing to modify
        if points.size == 0 or self.num_points <= 0:
            return points, {
                "num_points": 0,
                "scale_factor": self.scale_factor,
                "mode": self.mode,
            }

        # Compute bounding box
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (min_xyz + max_xyz) * 0.5
        bbox_size = (max_xyz - min_xyz) * self.scale_factor

        # Generate outlier positions
        if self.mode == "uniform":
            # Uniform sampling in expanded bounding box
            low = center - bbox_size * 0.5
            high = center + bbox_size * 0.5
            outliers = np.random.uniform(low, high, size=(self.num_points, 3))

        elif self.mode == "gaussian":
            # Gaussian noise around bounding box center
            outliers = np.random.normal(
                loc=center,
                scale=bbox_size * 0.2,
                size=(self.num_points, 3),
            )
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

        # Merge with original points
        new_points = np.concatenate([points, outliers], axis=0)

        metadata = {
            "num_points": self.num_points,
            "scale_factor": self.scale_factor,
            "mode": self.mode,
        }

        return new_points, metadata
