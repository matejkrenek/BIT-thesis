"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: augment.py
Responsibility: Wrapper dataset for applying augmentation transforms (synthetic defects) to point clouds on-the-fly, with optional caching.
"""

import hashlib
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class AugmentWrapperDataset(Dataset):
    """
    Wrapper dataset that applies a list of synthetic defects/augmentations to each sample of a base dataset.

    Args:
        dataset (Dataset): Base dataset (should provide Data with 'pos' attribute, ideally normalized).
        defects (list): List of defect transformations to apply (must have .apply() and .name).
        num_variants (int, optional): Number of augmented variants per sample (defaults to len(defects)).
        detailed (bool): If True, include defect logs in output Data.
        seed (int): Random seed for reproducibility.
        cache_npz_dir (str|os.PathLike|None): Optional directory for NPZ cache of defected samples.
        cache_read (bool): If True, read defected samples from NPZ cache when available.
        cache_write (bool): If True, save newly generated defected samples into NPZ cache.

    Output Data object fields:
        - original_pos: torch.FloatTensor (N, 3), original normalized point cloud
        - defected_pos: torch.FloatTensor (N, 3), defected point cloud
        - log: dict (if detailed=True), defect log
        - cache_path: str (if detailed=True and cache used), path to cached NPZ file
        - category, text: optional, passed through from base sample
    """

    def __init__(
        self,
        dataset: Dataset,
        defects: list = None,
        num_variants: int = None,
        detailed: bool = False,
        seed: int = 42,
        cache_npz_dir: str | os.PathLike | None = None,
        cache_read: bool = True,
        cache_write: bool = True,
    ):
        self.dataset = dataset
        self.defects = defects if defects is not None else []
        self.num_variants = len(self.defects) if num_variants is None else num_variants
        self.detailed = detailed
        self.seed = seed
        self.cache_npz_dir = Path(cache_npz_dir) if cache_npz_dir else None
        self.cache_read = bool(cache_read)
        self.cache_write = bool(cache_write)

        if self.cache_npz_dir is not None:
            self.cache_npz_dir.mkdir(parents=True, exist_ok=True)

        assert len(self.defects) > 0, "At least one defect must be provided"
        assert self.num_variants <= len(
            self.defects
        ), "num_variants must not exceed len(defects)"

    def _cache_path(self, base_idx: int, variant_id: int) -> Path | None:
        if self.cache_npz_dir is None:
            return None
        name = f"sample_{base_idx:07d}_variant_{variant_id:03d}_seed_{self.seed}.npz"
        return self.cache_npz_dir / name

    @staticmethod
    def _atomic_save_npz(path: Path, defected_pos: np.ndarray) -> None:
        """Atomically save a defected point cloud to a compressed NPZ file."""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, defected_pos=defected_pos)
        os.replace(tmp_path, path)

    def __len__(self) -> int:
        """Return the total number of samples (base samples × num_variants)."""
        return len(self.dataset) * self.num_variants

    def __getitem__(self, idx: int) -> Data | None:
        """
        Get a sample with a specific defect variant applied.

        Args:
            idx (int): Global index (across all base samples and variants)
        Returns:
            Data: torch_geometric.data.Data object with fields as described in class docstring.
        """
        base_idx = idx // self.num_variants
        variant_id = idx % self.num_variants
        data = self.dataset[base_idx]

        if not isinstance(data, Data) or not hasattr(data, "pos"):
            return None

        pos = data.pos

        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos).float()
        else:
            pos = pos.float()

        defect = self.defects[variant_id]
        sample_seed = self.seed + base_idx * self.num_variants + variant_id

        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)

        original_pos = pos.clone()
        cache_path = self._cache_path(base_idx=base_idx, variant_id=variant_id)
        defected_pos = None
        defect_log = {}

        if cache_path is not None and self.cache_read and cache_path.exists():
            with np.load(cache_path) as cached:
                defected_pos = np.asarray(cached["defected_pos"], dtype=np.float32)

        if defected_pos is None:
            pos_np = pos.numpy() if torch.is_tensor(pos) else np.asarray(pos)
            pos_np = pos_np.copy()
            defected_pos, defect_log = defect.apply(pos_np)
            defected_pos = np.asarray(defected_pos, dtype=np.float32)

            if cache_path is not None and self.cache_write:
                self._atomic_save_npz(cache_path, defected_pos)

        defected_pos_t = torch.from_numpy(defected_pos).float()
        output = Data(
            original_pos=original_pos,
            defected_pos=defected_pos_t,
        )
        category = getattr(data, "category", None)

        if category is not None:
            output.category = category

        text = getattr(data, "text", None)

        if text is not None:
            output.text = text

        if self.detailed:
            output.log = {defect.name: defect_log}

            if cache_path is not None:
                output.cache_path = str(cache_path)

        return output
