from .base import Defect
import numpy as np


class LocalDropout(Defect):
    """
    LocalDropout defect that removes points within specified local regions. Simulates small local
    holes typical for MVS caused by poor texture or reflective surfaces.
    """

    name: str = "local_dropout"

    def __init__(
        self, radius: float = 0.1, regions: int = 1, dropout_rate: float = 1.0
    ):
        """
        Args:
            radius (float): Radius of each local dropout region.
            regions (int): Number of local dropout regions to create.
            dropout_rate (float): Proportion of points to drop within each region (0.0 to 1.0).
        """
        self.radius = radius
        self.regions = regions
        self.dropout_rate = dropout_rate

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        # Select random center points for dropout regions
        region_indices = np.random.choice(
            points.shape[0], size=self.regions, replace=False
        )
        centers = points[region_indices]

        removed_mask = np.zeros(points.shape[0], dtype=bool)

        for center in centers:
            # Compute distance to center
            diff = points - center
            dist_sq = np.sum(diff * diff, axis=1)

            # Points inside radius
            in_region = dist_sq <= (self.radius * self.radius)

            # Optionally remove only a subset of points in the region
            if self.dropout_rate < 1.0:
                region_idxs = np.where(in_region)[0]
                if region_idxs.size > 0:
                    # Number of points to remove
                    k = int(np.ceil(region_idxs.size * self.dropout_rate))
                    selected = np.random.choice(region_idxs, size=k, replace=False)
                    mask = np.zeros_like(in_region)
                    mask[selected] = True
                    in_region = mask

            # Accumulate regions
            removed_mask |= in_region

        # Remove points
        remaining_points = points[~removed_mask]
        removed_count = int(removed_mask.sum())

        metadata = {
            "radius": self.radius,
            "regions": int(self.regions),
            "dropout_rate": self.dropout_rate,
            "removed_points": removed_count,
            "remaining_points": int(remaining_points.shape[0]),
        }

        return remaining_points, metadata
