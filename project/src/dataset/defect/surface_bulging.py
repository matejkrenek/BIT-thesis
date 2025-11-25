from .base import Defect
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SurfaceBulging(Defect):
    """
    Creates a local 'bulge' by inflating points outward from their centroid,
    simulating MVS depth-fusion overshoot where curved surfaces become
    exaggerated or swollen.

    Purpose:
        - Models convex distortions commonly produced in photogrammetry.
        - Encourages the model to recognize and correct outward deviation
          from the true shape.
        - Complements SurfaceFlattening (opposite curvature error).

    Parameters:
        radius (float):
            Radius of the affected region.

        strength (float):
            Controls how far points are pushed outward.

        falloff (float):
            Gaussian falloff controlling smoothness of the deformation.

        jitter (float):
            Small random noise added for realism.

        max_regions (int):
            Maximum number of bulged regions per sample.
    """

    name: str = "surface_bulging"

    def __init__(
        self,
        radius: float = 0.12,
        strength: float = 0.05,
        falloff: float = 2.0,
        jitter: float = 0.002,
        max_regions: int = 2,
    ):
        self.radius = float(radius)
        self.strength = float(strength)
        self.falloff = float(falloff)
        self.jitter = float(jitter)
        self.max_regions = int(max_regions)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0:
            return points, {"regions_used": 0}

        N = points.shape[0]

        modified = points.copy()

        # Choose deformation centers
        num_regions = np.random.randint(1, self.max_regions + 1)
        center_idxs = np.random.choice(N, size=num_regions, replace=False)
        centers = points[center_idxs]

        total_affected = 0

        for center in centers:
            diff = points - center
            dist = np.linalg.norm(diff, axis=1)
            mask = dist <= self.radius
            idxs = np.where(mask)[0]

            if idxs.size < 5:
                continue

            region_pts = points[idxs]

            # Compute region centroid
            c = region_pts.mean(axis=0)

            # Direction: from centroid outward
            dirs = region_pts - c
            dirs_norm = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
            normalized_dirs = dirs / dirs_norm

            # Gaussian falloff (smooth bulge)
            falloff_weights = np.exp(-(dist[idxs] ** 2) * self.falloff)

            # Bulge displacement
            displacement = normalized_dirs * (falloff_weights[:, None] * self.strength)

            deformed = region_pts + displacement

            # Add jitter
            if self.jitter > 0:
                deformed += np.random.normal(0.0, self.jitter, size=deformed.shape)

            modified[idxs] = deformed
            total_affected += idxs.size

        metadata = {
            "regions_used": num_regions,
            "total_points_bulged": total_affected,
            "radius": self.radius,
            "strength": self.strength,
            "falloff": self.falloff,
        }

        return modified, metadata
