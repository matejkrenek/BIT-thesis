from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from itertools import combinations
import numpy as np
from pytorch3d.ops import sample_farthest_points
import torch


def fps_subsample(points: np.ndarray, num_points: int):
    pts = torch.from_numpy(points).unsqueeze(0)  # (1, N, 3)
    sampled, _ = sample_farthest_points(pts, K=num_points)
    return sampled.squeeze(0).numpy()


class AugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, defects: list = [], num_variants: int = None):
        self.base = dataset
        self.combinations = self._generate_combinations(defects)
        self.num_variants = num_variants or len(self.combinations)

    def _generate_combinations(self, defects: list) -> list:
        combos = []
        for r in range(1, len(defects) + 1):
            for combo in combinations(defects, r):
                combos.append(list(combo))
        return combos

    def __len__(self):
        return len(self.base) * self.num_variants

    def __getitem__(self, idx):
        base_idx = idx // self.num_variants
        variant_id = idx % self.num_variants

        # Get the base data
        data = self.base[base_idx]
        original = data.pos.numpy()

        # Apply the defect chain
        defect_chain = self.combinations[variant_id]

        defected = original.copy()
        defected_log = {}

        for defect in defect_chain:
            defected, log = defect.apply(defected)
            defected_log[defect.name] = log

        # Normalize
        original_centroid = original.mean(axis=0)
        original_centered = original - original_centroid
        defected_centered = defected - original_centroid

        scale = np.max(np.linalg.norm(original_centered, axis=1))
        original = original_centered / scale
        defected = defected_centered / scale

        defected = fps_subsample(defected, data.pos.shape[0])

        return (original, defected)
