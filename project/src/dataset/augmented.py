from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from itertools import combinations
import numpy as np
import torch


class AugmentedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        defects: list = [],
        num_variants: int = None,
        detailed: bool = False,
    ):
        self.base = dataset
        self.defects = defects
        self.num_variants = len(defects) if num_variants is None else num_variants
        print(len(self.defects))
        self.detailed = detailed

    def __len__(self):
        return len(self.base) * self.num_variants

    def normalize_pc(self, points: np.ndarray):
        centroid = points.mean(axis=0)
        centered = points - centroid
        scale = np.max(np.linalg.norm(centered, axis=1))
        normalized = centered / scale
        return normalized, centroid, scale

    def __getitem__(self, idx):
        base_idx = idx // self.num_variants
        variant_id = idx % self.num_variants

        # Get the base data
        data = self.base[base_idx]

        if not isinstance(data, Data) or not hasattr(data, "pos"):
            return None  # vadný vzorek

        original = data.pos.numpy()

        original_norm, centroid, scale = self.normalize_pc(original)

        defect = self.defects[variant_id]

        torch.manual_seed(idx)
        np.random.seed(idx)

        defected_norm = original_norm.copy()
        defected_log = {}

        defected_norm, log = defect.apply(defected_norm)
        defected_log[defect.name] = log

        original_t = torch.from_numpy(original_norm).float()
        defected_t = torch.from_numpy(defected_norm).float()

        return (
            (original_t, defected_t)
            if not self.detailed
            else (original_t, defected_t, defected_log)
        )
