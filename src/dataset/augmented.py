"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: augmented.py
Responsibility: Dataset wrapper for applying a list of synthetic defects/augmentations to each sample of a base 3D point cloud dataset.
Deprecated: This class is deprecated in favor of the more advanced AugmentWrapperDataset in src/dataset/wrapper/augment.py
"""

from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import torch


class AugmentedDataset(Dataset):
    """
    Dataset wrapper that applies a list of synthetic defects/augmentations to each sample of a base dataset.

    Args:
        dataset: Base dataset to wrap (must yield torch_geometric.data.Data with 'pos').
        defects: List of defect objects with .apply() method and .name attribute.
        num_variants: Number of variants per sample (default: len(defects)).
        detailed: If True, returns detailed logs for each sample.

    Returns:
        If detailed is False:
            (original_t, defected_t)
        If detailed is True:
            (original_t, defected_t, defected_log)
        where original_t and defected_t are torch.FloatTensor (N, 3), defected_log is a dict.
    """

    def __init__(
        self,
        dataset: Dataset,
        defects: list = None,
        num_variants: int = None,
        detailed: bool = False,
    ):
        """
        Args:
            dataset: Base dataset to wrap.
            defects: List of defect objects with .apply() method and .name attribute.
            num_variants: Number of variants per sample (default: len(defects)).
            detailed: If True, returns detailed logs for each sample.
        """
        self.base = dataset
        self.defects = defects if defects is not None else []
        self.num_variants = len(self.defects) if num_variants is None else num_variants
        self.detailed = detailed

    def __len__(self):
        return len(self.base) * self.num_variants

    @staticmethod
    def normalize_pc(points: np.ndarray):
        centroid = points.mean(axis=0)
        centered = points - centroid
        scale = np.max(np.linalg.norm(centered, axis=1))
        normalized = centered / (scale + 1e-8)
        return normalized, centroid, scale

    def __getitem__(self, idx):
        base_idx = idx // self.num_variants
        variant_id = idx % self.num_variants

        data = self.base[base_idx]

        if not isinstance(data, Data) or not hasattr(data, "pos"):
            return None

        original = (
            data.pos.cpu().numpy() if hasattr(data.pos, "cpu") else np.asarray(data.pos)
        )
        original_norm, centroid, scale = self.normalize_pc(original)
        defect = self.defects[variant_id]

        # Seed
        torch.manual_seed(idx)
        np.random.seed(idx)

        defected_norm = original_norm.copy()
        defected_log = {}
        defected_norm, log = defect.apply(defected_norm)
        defected_log[defect.name] = log
        original_t = torch.from_numpy(original_norm).float()
        defected_t = torch.from_numpy(defected_norm).float()

        if not self.detailed:
            return (original_t, defected_t)
        else:
            return (original_t, defected_t, defected_log)
