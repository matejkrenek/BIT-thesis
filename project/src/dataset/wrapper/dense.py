"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: dense.py
Responsibility: Wrapper dataset for converting sparse point clouds to dense representations using mesh sampling and optional memory mapping.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing as mp
from torch_geometric.data import Data


class DenseWrapperDataset(Dataset):
    """
    Wrapper dataset that converts sparse point clouds to dense representations by sampling points from mesh surfaces.

    Args:
        dataset (Dataset): Base dataset (must provide mesh_pos and face attributes).
        root (str): Directory to store dense point cloud .npz files.
        num_points (int): Number of dense points to sample per mesh.
        use_mmap (bool): Whether to use memory mapping for loading dense .npz files.
    """

    def __init__(
        self,
        dataset: Dataset,
        root: str,
        num_points: int = 100_000,
        use_mmap: bool = True,
    ):
        self.dataset = dataset
        self.root = root
        self.num_points = num_points
        self.use_mmap = use_mmap

    def __len__(self) -> int:
        """Return the number of samples in the base dataset."""
        return len(self.dataset)

    def _get_dense_path(self, idx: int) -> str:
        """Get the path to the dense .npz file for a given sample index."""
        filename = self.dataset.files[idx]
        name = os.path.splitext(filename)[0]
        return os.path.join(self.root, f"{name}.npz")

    def _load_dense(self, path: str) -> np.ndarray:
        """Load dense points from a .npz file, optionally using memory mapping."""
        if self.use_mmap:
            data = np.load(path, mmap_mode="r")["points"]
        else:
            data = np.load(path)["points"]
        return data

    def _sample_dense(self, idx: int, overwrite: bool = False) -> None:
        """
        Sample dense points from the mesh and save to .npz file if not already present.
        Args:
            idx (int): Index of the sample in the base dataset.
            overwrite (bool): If True, overwrite existing .npz file.
        """
        os.makedirs(self.root, exist_ok=True)
        data = self.dataset[idx]
        mesh_pos = data.mesh_pos
        faces = data.face.t()
        dense_path = self._get_dense_path(idx)

        if os.path.exists(dense_path) and not overwrite:
            return

        from pytorch3d.ops import sample_points_from_meshes
        from pytorch3d.structures import Meshes

        mesh = Meshes(verts=[mesh_pos], faces=[faces])
        points = sample_points_from_meshes(
            mesh,
            num_samples=self.num_points,
            return_normals=False,
            return_textures=False,
        )
        np.savez(dense_path, points=points.cpu().numpy())

    def __getitem__(self, idx: int) -> Data | None:
        """
        Get a sample with dense points loaded from .npz file, replacing the 'pos' attribute.
        Args:
            idx (int): Index of the sample in the base dataset.
        Returns:
            Data: torch_geometric.data.Data object with dense points in 'pos', or None if invalid.
        """
        data = self.dataset[idx]

        if not isinstance(data, Data) or not hasattr(data, "pos"):
            return None

        self._sample_dense(idx)
        dense_path = self._get_dense_path(idx)
        dense_points = self._load_dense(dense_path)
        dense_points = torch.from_numpy(np.asarray(dense_points)).float().squeeze(0)
        data.pos = dense_points

        return data
