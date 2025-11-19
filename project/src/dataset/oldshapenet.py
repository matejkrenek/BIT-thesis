from torch.utils.data import Dataset, DataLoader
from .config import DatasetConfig
from .downloader.base import BaseDownloader
from zipfile import ZipFile
from logger import logger
from .sample import DatasetSample
import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np


class ShapeNetDataset(Dataset):
    """ShapeNet dataset class."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    def _get_obj_files(self) -> list:
        """Get a list of all OBJ files in the dataset directory."""
        if not hasattr(self, "_obj_files_cache"):
            self._obj_files_cache = list(self.config.local_dir.glob("**/*.obj"))
        return self._obj_files_cache

    def download(self, downloader: BaseDownloader) -> bool:
        """
        Download and prepare the ShapeNet dataset.
        Args:
            downloader (BaseDownloader): The downloader instance to use.
        Returns:
            bool: True if download and preparation were successful, False otherwise.
        """
        if self.config.local_dir.exists() and any(
            self.config.local_dir.glob("**/*.obj")
        ):
            logger.info(
                "Dataset already exists and contains OBJ files. Skipping download."
            )
            return True

        download = downloader.download()

        # Extract downloaded zip files and clean up non-OBJ files
        for zip_path in downloader.config.local_dir.glob("*.zip"):
            extract_dir = self.config.local_dir

            with ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            # Remove non-OBJ files
            for path in extract_dir.glob(zip_path.stem + "/**/*"):
                if path.is_file() and path.suffix.lower() != ".obj":
                    path.unlink()

            # Remove empty directories
            for path in sorted(extract_dir.glob(zip_path.stem + "/**/*"), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()

            zip_path.unlink()

        return download

    def loader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader for the dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: batch,
        )

    def _load_mesh(self, obj_path: str) -> o3d.geometry.TriangleMesh:
        """Load a 3D mesh from an OBJ file."""
        mesh = o3d.io.read_triangle_mesh(str(obj_path))

        mesh.compute_vertex_normals()

        return mesh

    def _mesh_to_pcn(
        self, mesh: o3d.geometry.TriangleMesh, num_points: int
    ) -> o3d.geometry.PointCloud:
        """Sample a point cloud from the mesh surface."""
        pcn = mesh.sample_points_uniformly(number_of_points=num_points)
        return pcn

    def _pcn_to_mesh(self, pcn: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Reconstruct a mesh from the point cloud using Poisson reconstruction."""
        disntance = pcn.compute_nearest_neighbor_distance()
        avg_dist = np.mean(disntance)
        radius = 3 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcn, o3d.utility.DoubleVector([radius, radius * 2])
        )

        return mesh

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._get_obj_files())

    def __getitem__(self, idx) -> DatasetSample:
        """Get a dataset sample by index."""
        obj_path = self._get_obj_files()[idx]
        mesh = self._load_mesh(obj_path)
        original_pcn = self._mesh_to_pcn(mesh, num_points=40000)
        syntetic_pcn = original_pcn
        pcn_mesh = self._pcn_to_mesh(original_pcn)

        return DatasetSample(
            obj_path=obj_path,
            mesh=mesh,
            original_pcn=original_pcn,
            syntetic_pcn=syntetic_pcn,
            pcn_mesh=pcn_mesh,
            defects={},
        )
