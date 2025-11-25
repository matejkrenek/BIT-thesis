from .base import Defect
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SurfaceFlattening(Defect):
    """
    Locally flattens a region of the point cloud by projecting points onto
    a tangent plane estimated from PCA. Simulates MVS depth-fusion smoothing
    that collapses real curvature into flat surfaces.

    Purpose:
        - Models loss of curvature on low-texture or low-confidence areas.
        - Trains the model to restore curved surfaces.
        - Complements defects like HairNoise and Bridging.

    Parameters:
        radius (float):
            Radius of the local region to be flattened.

        plane_jitter (float):
            Adds small noise after projection for realism.

        max_regions (int):
            Number of flattened areas to apply per sample.
    """

    name: str = "surface_flattening"

    def __init__(
        self,
        radius: float = 0.1,
        plane_jitter: float = 0.002,
        max_regions: int = 2,
    ):
        self.radius = float(radius)
        self.plane_jitter = float(plane_jitter)
        self.max_regions = int(max_regions)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0:
            return points, {"regions_used": 0}

        N = points.shape[0]

        # Choose random centers
        n_regions = np.random.randint(1, self.max_regions + 1)
        center_idx = np.random.choice(N, size=n_regions, replace=False)
        centers = points[center_idx]

        # kNN search for local neighborhood
        nbrs = NearestNeighbors(n_neighbors=40, algorithm="auto").fit(points)
        distances, indices = nbrs.kneighbors(points)

        modified_points = points.copy()
        total_affected = 0

        for center in centers:
            # Compute distances to all points (not using neighbors first)
            diff = points - center
            dist_sq = np.sum(diff * diff, axis=1)

            # Mask points in radius
            mask = dist_sq <= (self.radius * self.radius)
            region_pts_idx = np.where(mask)[0]

            if region_pts_idx.size < 5:
                continue

            region_pts = points[region_pts_idx]

            # Estimate tangent plane using PCA eigenvectors
            local_center = region_pts.mean(axis=0)
            cov = np.cov((region_pts - local_center).T)
            eigvals, eigvecs = np.linalg.eigh(cov)

            # The two largest eigenvectors span the plane
            u = eigvecs[:, 2]
            v = eigvecs[:, 1]

            # Project points onto the plane
            proj = local_center + np.dot(
                region_pts - local_center, np.vstack([u, v]).T
            ) @ np.vstack([u, v])

            # Add small jitter
            if self.plane_jitter > 0:
                proj += np.random.normal(0.0, self.plane_jitter, size=proj.shape)

            # Update modified points
            modified_points[region_pts_idx] = proj
            total_affected += region_pts_idx.size

        metadata = {
            "regions_used": n_regions,
            "total_points_flattened": total_affected,
            "radius": self.radius,
            "plane_jitter": self.plane_jitter,
        }

        return modified_points, metadata
