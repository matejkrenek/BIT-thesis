from pathlib import Path
from loguru import logger
import open3d as o3d
import numpy as np

from .mesh_loader import MeshLoader
from .point_sampler import PointSampler


class PointCloudBuilder:
    def __init__(
        self,
        extracted_dir: str = "data/extracted",
        output_dir: str = "data/processed",
        num_points: int = 16384,
    ):
        self.extracted_dir = Path(extracted_dir)
        self.output_dir = Path(output_dir)
        self.sampler = PointSampler(num_points)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_synset(self, synset_id: str):
        synset_path = self.extracted_dir / synset_id
        out_synset_dir = self.output_dir / synset_id
        out_synset_dir.mkdir(parents=True, exist_ok=True)

        loader = MeshLoader(synset_path)

        for mesh, model_id in loader.load_meshes():
            pts = self.sampler.sample(mesh)
            self._save_pointcloud(out_synset_dir, model_id, pts)

    def _save_pointcloud(self, out_synset_dir: Path, model_id: str, points: np.ndarray):
        out_path = out_synset_dir / f"{model_id}.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(out_path), pcd)
        logger.info(f"Saved pointcloud: {out_path}")
