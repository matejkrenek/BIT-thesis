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
        self.combinations = self._generate_combinations(defects)
        self.num_variants = num_variants or len(self.combinations)
        self.detailed = detailed

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

        if not isinstance(data, Data) or not hasattr(data, "pos"):
            return None  # vadný vzorek

        original = data.pos.numpy()

        # Apply the defect chain
        defect_chain = self.combinations[variant_id]

        torch.manual_seed(idx)
        np.random.seed(idx)

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
        original = torch.from_numpy(original_centered / scale).float()
        defected = torch.from_numpy(defected_centered / scale).float()
        
        return (
            (original, defected)
            if not self.detailed
            else (original, defected, defected_log)
        )
