from .base import Defect
import numpy as np


class AnisotropicStretchNoise(Defect):
    """
    Applies local anisotropic stretching to the point cloud,
    simulating MVS directional distortion where geometry becomes
    slanted, elongated, or warped along one axis.

    Purpose:
        - Models realistic photogrammetry errors caused by
          depth inconsistency or biased viewing angles.
        - Encourages the model to recognize and correct
          directional shape distortions.

    Parameters:
        radius (float):
            Region of influence for the transformation.

        stretch_factor (float):
            Maximum amount of stretching along the primary axis.

        max_regions (int):
            How many regions can be stretched.

        jitter (float):
            Optional random noise added after stretching for realism.
    """

    name: str = "anisotropic_stretch_noise"

    def __init__(
        self,
        radius: float = 0.15,
        stretch_factor: float = 0.25,
        max_regions: int = 2,
        jitter: float = 0.002,
    ):
        self.radius = float(radius)
        self.stretch_factor = float(stretch_factor)
        self.max_regions = int(max_regions)
        self.jitter = float(jitter)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0:
            return points, {"regions_used": 0}

        N = points.shape[0]
        modified = points.copy()

        # Number of distorted regions
        num_regions = np.random.randint(1, self.max_regions + 1)
        region_centers = points[np.random.choice(N, size=num_regions, replace=False)]

        total_affected = 0

        for center in region_centers:
            diff = points - center
            dist = np.linalg.norm(diff, axis=1)
            mask = dist <= self.radius
            idxs = np.where(mask)[0]

            if idxs.size < 5:
                continue

            region_pts = points[idxs]

            # Random primary axis of stretching
            direction = np.random.normal(0.0, 1.0, size=3)
            direction /= np.linalg.norm(direction) + 1e-8

            # Build anisotropic transform
            s = np.random.uniform(0.1, 1.0) * self.stretch_factor
            stretch_matrix = np.eye(3) + s * np.outer(direction, direction)

            # Gaussian falloff
            falloff = np.exp(-(dist[idxs] ** 2) / (self.radius**2))

            # Apply transformation
            region_center = region_pts.mean(axis=0)
            centered = region_pts - region_center
            transformed = centered @ stretch_matrix.T + region_center

            # Apply falloff
            transformed = (
                region_pts * (1 - falloff[:, None]) + transformed * falloff[:, None]
            )

            # Add jitter
            if self.jitter > 0:
                transformed += np.random.normal(
                    0.0, self.jitter, size=transformed.shape
                )

            modified[idxs] = transformed
            total_affected += idxs.size

        metadata = {
            "regions_used": num_regions,
            "total_points_stretched": total_affected,
            "radius": self.radius,
            "stretch_factor": self.stretch_factor,
        }

        return modified, metadata
