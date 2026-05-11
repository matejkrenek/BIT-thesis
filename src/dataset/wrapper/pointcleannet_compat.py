"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: pointcleannet_compat.py
Responsibility: Wrapper dataset for extracting PointCleanNet-compatible patches from point clouds.

Attribution:
    This compatibility wrapper is adapted from PointCleanNet dataset implementation:
    https://github.com/mrakotosaon/pointcleannet/blob/master/noise_removal/dataset.py
    The adaptation keeps the patch extraction behavior while removing unused parts and
    integrating the code with this repository's dataset interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.spatial as spatial
import torch
import torch.utils.data as data
from torch_geometric.data import Data


@dataclass
class _Shape:
    pts: np.ndarray
    kdtree: spatial.cKDTree
    clean_points: np.ndarray | None = None
    clean_kdtree: spatial.cKDTree | None = None


class _Cache:
    def __init__(self, capacity: int, loader, loadfunc):
        self.elements: dict[int, Any] = {}
        self.used_at: dict[int, int] = {}
        self.capacity = int(capacity)
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id: int):
        if element_id not in self.elements:
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1
        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):
    """
    Wrapper dataset for extracting PointCleanNet-compatible patches from point clouds.

    Output tuple format (for patch_features=["original"]):
        (patch_points, original_center_point, patch_radius_scalar, pca_transform)

    Args:
        dataset: Base dataset.
        patch_radius: Radius for patch extraction.
        points_per_patch: Number of points per patch.
        patch_features: Features to extract for each patch.
        seed: Optional random seed for reproducibility.
        identical_epochs: If True, patches are identical across epochs.
        use_pca: Whether to compute PCA transform for each patch.
        center: Centering mode for patches.
    """

    def __init__(
        self,
        dataset,
        patch_radius,
        points_per_patch: int,
        patch_features,
        seed: int | None = None,
        identical_epochs: bool = False,
        use_pca: bool = True,
        center: str = "point",
        point_tuple: int = 1,
        cache_capacity: int = 1,
        point_count_std: float = 0.0,
        sparse_patches: bool = False,
        shape_names: list[str] | None = None,
    ):
        self.dataset = dataset
        self.patch_features = list(patch_features)
        self.patch_radius = list(patch_radius)
        self.points_per_patch = int(points_per_patch)
        self.identical_epochs = bool(identical_epochs)
        self.use_pca = bool(use_pca)
        self.center = str(center)
        self.point_tuple = int(point_tuple)
        self.point_count_std = float(point_count_std)
        self.sparse_patches = bool(sparse_patches)

        if self.point_tuple != 1:
            raise ValueError(
                "point_tuple > 1 is not supported by compatibility wrapper"
            )
        if self.sparse_patches:
            raise ValueError(
                "sparse_patches=True is not supported by compatibility wrapper"
            )
        if self.center not in {"point", "mean", "none"}:
            raise ValueError("center must be one of {'point', 'mean', 'none'}")

        self.include_clean_points = "clean_points" in self.patch_features
        self.include_original = "original" in self.patch_features

        supported_features = {"clean_points", "original"}
        unknown = [f for f in self.patch_features if f not in supported_features]
        if unknown:
            raise ValueError(f"Unsupported patch features: {unknown}")

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.shape_names = (
            shape_names
            if shape_names is not None
            else [f"shape_{i}" for i in range(len(dataset))]
        )
        if len(self.shape_names) != len(dataset):
            raise ValueError("shape_names length must match wrapped dataset length")

        self.shape_cache = _Cache(
            max(int(cache_capacity), 1),
            self,
            PointcloudPatchDataset.load_shape_by_index,
        )

        self.shape_patch_count: list[int] = []
        self.patch_radius_absolute: list[list[float]] = []
        for shape_ind, _ in enumerate(self.shape_names):
            shape = self.shape_cache.get(shape_ind)
            self.shape_patch_count.append(int(shape.pts.shape[0]))
            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            self.patch_radius_absolute.append(
                [bbdiag * rad for rad in self.patch_radius]
            )

    def __len__(self):
        return int(sum(self.shape_patch_count))

    def shape_index(self, index: int):
        shape_patch_offset = 0
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if shape_patch_offset <= index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                return shape_ind, shape_patch_ind
            shape_patch_offset += shape_patch_count
        raise IndexError(f"Global patch index out of range: {index}")

    @staticmethod
    def _extract_cloud_pair(sample) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(sample, Data):
            if hasattr(sample, "defected_pos"):
                defected = sample.defected_pos
            elif hasattr(sample, "pos"):
                defected = sample.pos
            else:
                raise ValueError("Data sample missing defected_pos/pos")

            if hasattr(sample, "original_pos"):
                original = sample.original_pos
            else:
                original = defected

        elif isinstance(sample, dict):
            defected = sample.get("defected_pos", sample.get("pos", None))
            original = sample.get("original_pos", defected)
            if defected is None:
                raise ValueError("Dict sample missing defected_pos/pos")
        else:
            raise TypeError(f"Unsupported sample type: {type(sample).__name__}")

        defected_np = np.asarray(
            torch.as_tensor(defected).float().cpu().numpy(), dtype=np.float32
        )
        original_np = np.asarray(
            torch.as_tensor(original).float().cpu().numpy(), dtype=np.float32
        )

        if defected_np.ndim != 2 or defected_np.shape[1] != 3:
            raise ValueError(
                f"defected cloud must have shape (N,3), got {defected_np.shape}"
            )
        if original_np.ndim != 2 or original_np.shape[1] != 3:
            raise ValueError(
                f"original cloud must have shape (N,3), got {original_np.shape}"
            )

        return defected_np, original_np

    def load_shape_by_index(self, shape_ind: int):
        sample = self.dataset[shape_ind]
        defected_np, original_np = self._extract_cloud_pair(sample)

        # Match official behavior where patch centers and query neighbors come from noisy/defected cloud.
        sys_rec_limit = int(max(1000, round(defected_np.shape[0] / 10)))
        try:
            import sys

            sys.setrecursionlimit(sys_rec_limit)
        except Exception:
            pass

        kdtree = spatial.cKDTree(defected_np, 10)
        clean_kdtree = None
        if self.include_clean_points:
            clean_kdtree = spatial.cKDTree(original_np, 10)

        return _Shape(
            pts=defected_np,
            kdtree=kdtree,
            clean_points=original_np,
            clean_kdtree=clean_kdtree,
        )

    def _select_patch_points(
        self,
        patch_radius: float,
        global_point_index: int,
        center_point_ind: int,
        shape: _Shape,
        radius_index: int,
        scale_ind_range: np.ndarray,
        patch_pts_valid: list[int],
        patch_pts: torch.Tensor,
        clean_points: bool = False,
    ):
        if clean_points:
            if shape.clean_kdtree is None or shape.clean_points is None:
                raise ValueError(
                    "clean_points feature requested but clean cloud is missing"
                )
            patch_point_inds = np.array(
                shape.clean_kdtree.query_ball_point(
                    shape.pts[center_point_ind, :], patch_radius
                )
            )
            points_base = shape.clean_points
        else:
            patch_point_inds = np.array(
                shape.kdtree.query_ball_point(
                    shape.pts[center_point_ind, :], patch_radius
                )
            )
            points_base = shape.pts

        if self.identical_epochs:
            self.rng.seed((self.seed + global_point_index) % (2**32))

        point_count = min(self.points_per_patch, len(patch_point_inds))

        if self.point_count_std > 0:
            ratio = self.rng.uniform(1.0 - self.point_count_std * 2.0, 1.0)
            point_count = max(5, round(point_count * ratio))
            point_count = min(point_count, len(patch_point_inds))

        if point_count < len(patch_point_inds):
            patch_point_inds = patch_point_inds[
                self.rng.choice(len(patch_point_inds), point_count, replace=False)
            ]

        start = radius_index * self.points_per_patch
        end = start + point_count
        scale_ind_range[radius_index, :] = [start, end]
        patch_pts_valid += list(range(start, end))

        if len(patch_point_inds) > 0:
            patch_pts[start:end, :] = torch.from_numpy(points_base[patch_point_inds, :])

        if self.center == "mean":
            if end > start:
                patch_pts[start:end, :] = patch_pts[start:end, :] - patch_pts[
                    start:end, :
                ].mean(0)
        elif self.center == "point":
            patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(
                shape.pts[center_point_ind, :]
            )
        elif self.center == "none":
            pass

        patch_pts[start:end, :] = patch_pts[start:end, :] / max(patch_radius, 1e-8)
        return patch_pts, patch_pts_valid, scale_ind_range

    def __getitem__(self, index: int):
        shape_ind, patch_ind = self.shape_index(int(index))
        shape = self.shape_cache.get(shape_ind)

        center_point_ind = patch_ind

        patch_pts = torch.zeros(
            self.points_per_patch * len(self.patch_radius_absolute[shape_ind]), 3
        ).float()
        patch_pts_valid: list[int] = []
        scale_ind_range = np.zeros(
            [len(self.patch_radius_absolute[shape_ind]), 2], dtype="int"
        )

        for radius_index, patch_radius in enumerate(
            self.patch_radius_absolute[shape_ind]
        ):
            patch_pts, patch_pts_valid, scale_ind_range = self._select_patch_points(
                patch_radius,
                int(index),
                center_point_ind,
                shape,
                radius_index,
                scale_ind_range,
                patch_pts_valid,
                patch_pts,
                clean_points=False,
            )

        patch_clean_points = None
        if self.include_clean_points:
            patch_clean_points = torch.zeros(self.points_per_patch, 3).float()
            tmp_valid: list[int] = []
            clean_scale_range = np.zeros(
                [len(self.patch_radius_absolute[shape_ind]), 2], dtype="int"
            )
            clean_patch_radius = self.patch_radius_absolute[shape_ind][0]
            patch_clean_points, _, _ = self._select_patch_points(
                clean_patch_radius,
                int(index),
                center_point_ind,
                shape,
                0,
                clean_scale_range,
                tmp_valid,
                patch_clean_points,
                clean_points=True,
            )

        if self.use_pca and len(patch_pts_valid) > 0:
            pts_mean = patch_pts[patch_pts_valid, :].mean(0)
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - pts_mean

            trans, _, _ = torch.svd(torch.t(patch_pts[patch_pts_valid, :]))
            patch_pts[patch_pts_valid, :] = torch.mm(
                patch_pts[patch_pts_valid, :], trans
            )

            cp_new = -pts_mean
            cp_new = torch.matmul(cp_new, trans)
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - cp_new
        else:
            trans = torch.eye(3).float()

        original_center = torch.from_numpy(shape.pts[center_point_ind, :]).float()
        patch_radius_scalar = float(self.patch_radius_absolute[shape_ind][0])

        patch_feats = ()
        for pfeat in self.patch_features:
            if pfeat == "clean_points":
                if patch_clean_points is None:
                    raise RuntimeError("clean_points patch requested but not prepared")
                patch_feats = patch_feats + (patch_clean_points,)
            elif pfeat == "original":
                patch_feats = patch_feats + (original_center, patch_radius_scalar)
        return (patch_pts,) + patch_feats + (trans,)
