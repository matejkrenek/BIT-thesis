from .base import Defect
import numpy as np


class LargeMissingRegion(Defect):
    """
    Removes a large structural region of the point cloud by slicing along a
    random direction. Guaranteed removal of a specified fraction of points.

    Purpose:
        - Simulates large missing chunks typical for MVS failures.
        - Always removes a continuous part of the object (unlike sphere dropout).
        - Encourages global shape reasoning during reconstruction.

    Parameters:
        removal_fraction (float):
            Fraction of points to remove in [0.1, 0.7].
            Example: 0.3 removes the top 30% along a random view direction.
    """

    name: str = "large_missing_region"

    def __init__(self, removal_fraction: float = 0.3):
        self.removal_fraction = float(removal_fraction)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0:
            return points, {
                "removal_fraction": self.removal_fraction,
                "removed_points": 0,
                "remaining_points": 0,
            }

        # Random direction
        normal = np.random.normal(0.0, 1.0, size=3)
        normal /= np.linalg.norm(normal) + 1e-8

        # Project onto normal
        proj = points @ normal

        # Percentile threshold
        cutoff = np.percentile(proj, 100.0 * (1.0 - self.removal_fraction))

        # Remove all points above threshold
        mask = proj <= cutoff
        remaining = points[mask]
        removed = points[~mask]

        metadata = {
            "removal_fraction": self.removal_fraction,
            "removed_points": int(removed.shape[0]),
            "remaining_points": int(remaining.shape[0]),
        }

        return remaining, metadata
