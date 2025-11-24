from .base import BaseViewer
import polyscope as ps
import numpy as np


class PointCloudViewer(BaseViewer):
    """Viewer for point cloud data."""

    def __init__(self):
        super().__init__()

        if not self.initialized:
            ps.init()
            self.initialized = True

    def show(
        self, points: np.ndarray, name="pointcloud", radius=0.01, color=(1.0, 0.0, 0.0)
    ):
        ps.register_point_cloud(name, points, radius=radius, color=color)
        ps.show()

    def clear(self):
        ps.remove_all_structures()
