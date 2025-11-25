from .base import Defect
import numpy as np


class FloatingCluster(Defect):
    """
    FloatingClusters adds small clusters of points detached from the main object bounding box.
    """

    name: str = "floating_cluster"

    def __init__(
        self,
        cluster_radius: float = 0.05,
        clusters: int = 1,
        points_per_cluster: int = 50,
        offset_factor: float = 1.5,
    ):
        """
        Args:
            cluster_radius (float): Radius of each floating cluster.
            clusters (int): Number of floating clusters to create.
            points_per_cluster (int): Number of points in each floating cluster.
            offset_factor (float): Factor to determine how far clusters are from the main bounding box.
        """
        self.cluster_radius = cluster_radius
        self.clusters = clusters
        self.points_per_cluster = points_per_cluster
        self.offset_factor = offset_factor

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        # Compute bounding box of the original points
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        bbox_size = max_xyz - min_xyz
        bbox_center = (min_xyz + max_xyz) * 0.5

        all_clusters = []

        for _ in range(self.clusters):
            # Randomly calculate where to place the cluster
            direction = np.random.normal(0.0, 1.0, size=3)
            direction /= np.linalg.norm(direction) + 1e-8

            offset_mag = np.random.uniform(
                self.offset_factor * 0.5, self.offset_factor * 1.5
            )

            cluster_center = bbox_center + direction * offset_mag * bbox_size

            # Generate points in a sphere around the cluster center
            pts = np.random.normal(0.0, 1.0, size=(self.points_per_cluster, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
            pts *= np.random.rand(self.points_per_cluster, 1) * self.cluster_radius

            # Offset points to cluster center
            pts += cluster_center
            all_clusters.append(pts)

        new_points = np.concatenate([points] + all_clusters, axis=0)

        metadata = {
            "clusters_used": self.clusters,
            "cluster_radius": self.cluster_radius,
            "points_per_cluster": self.points_per_cluster,
            "offset_factor": self.offset_factor,
        }

        return new_points, metadata
