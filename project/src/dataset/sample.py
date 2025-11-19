from dataclasses import dataclass, field
from pathlib import Path
import open3d as o3d
from typing import Optional


@dataclass
class DatasetSample:
    obj_path: Path
    mesh: o3d.geometry.TriangleMesh
    original_pcn: o3d.geometry.PointCloud
    syntetic_pcn: o3d.geometry.PointCloud
    defects: dict
    pcn_mesh: o3d.geometry.TriangleMesh = None
