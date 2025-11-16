import open3d as o3d
import numpy as np
from pathlib import Path


def load_ply(path: str | Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)
