import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing as mp
from torch_geometric.data import Data


class DenseWrapperDataset(Dataset):
    def __init__(
        self,
        dataset,
        root: str,
        num_points: int = 100_000,
        use_mmap: bool = True,
    ):
        self.dataset = dataset
        self.root = root
        self.num_points = num_points
        self.use_mmap = use_mmap

    def __len__(self):
        return len(self.dataset)

    def _get_dense_path(self, idx):
        filename = self.dataset.files[idx]
        name = os.path.splitext(filename)[0]
        return os.path.join(self.root, f"{name}.npz")

    def _load_dense(self, path):
        if self.use_mmap:
            data = np.load(path, mmap_mode="r")["points"]
        else:
            data = np.load(path)["points"]

        return data

    def _sample_dense(self, idx, overwrite=False):
        os.makedirs(self.root, exist_ok=True)

        data = self.dataset[idx]
        mesh_pos = data.mesh_pos
        faces = data.face.t()
        dense_path = self._get_dense_path(idx)

        if os.path.exists(dense_path) and not overwrite:
            return

        # Sample points from the mesh
        from pytorch3d.ops import sample_points_from_meshes
        from pytorch3d.structures import Meshes

        # Wrap mesh data in Meshes object
        mesh = Meshes(verts=[mesh_pos], faces=[faces])

        points = sample_points_from_meshes(
            mesh,
            num_samples=self.num_points,
            return_normals=False,
            return_textures=False,
        )

        # Save the sampled points
        np.savez(dense_path, points=points.cpu().numpy())

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if not isinstance(data, Data) or not hasattr(data, "pos"):
            return None  # vadný vzorek
        
        self._sample_dense(idx)
        dense_path = self._get_dense_path(idx)
        dense_points = self._load_dense(dense_path)
        dense_points = torch.from_numpy(np.asarray(dense_points)).float().squeeze(0)
        data.pos = dense_points

        return data
