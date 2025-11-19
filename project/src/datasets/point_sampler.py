import numpy as np
import open3d as o3d


class PointSampler:
    def __init__(self, num_points: int = 16384):
        self.num_points = num_points

    def normalize(self, mesh: o3d.geometry.TriangleMesh):
        mesh = mesh.translate(-mesh.get_center())
        scale = max(mesh.get_max_bound() - mesh.get_min_bound())
        mesh.scale(1.0 / scale, center=(0, 0, 0))
        return mesh

    def sample(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        mesh = self.normalize(mesh)
        pcd = mesh.sample_points_uniformly(number_of_points=self.num_points)
        return np.asarray(pcd.points)
