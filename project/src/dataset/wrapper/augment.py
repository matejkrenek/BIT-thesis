import hashlib
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class AugmentWrapperDataset(Dataset):
    """
    Augmentation wrapper that applies defects to point clouds.

    Expects the input dataset to already have normalized point clouds.
    Each base sample is replicated num_variants times with different defects applied.
    """

    def __init__(
        self,
        dataset: Dataset,
        defects: list = [],
        num_variants: int = None,
        detailed: bool = False,
        seed: int = 42,
        cache_npz_dir: str | os.PathLike | None = None,
        cache_read: bool = True,
        cache_write: bool = True,
    ):
        """
        Args:
            dataset: Base dataset (should provide Data with 'pos' attribute)
            defects: List of defect transformations to apply
            num_variants: Number of augmented variants per sample (defaults to len(defects))
            detailed: If True, include defect logs in output
            seed: Random seed for reproducibility
            cache_npz_dir: Optional directory for defected sample cache in NPZ files
            cache_read: If True, read defected samples from NPZ cache when available
            cache_write: If True, save newly generated defected samples into NPZ cache
        """
        self.dataset = dataset
        self.defects = defects
        self.num_variants = len(defects) if num_variants is None else num_variants
        self.detailed = detailed
        self.seed = seed
        self.cache_npz_dir = Path(cache_npz_dir) if cache_npz_dir else None
        self.cache_read = bool(cache_read)
        self.cache_write = bool(cache_write)

        if self.cache_npz_dir is not None:
            self.cache_npz_dir.mkdir(parents=True, exist_ok=True)

        assert len(defects) > 0, "At least one defect must be provided"
        assert self.num_variants <= len(
            defects
        ), "num_variants must not exceed len(defects)"

    def _cache_path(self, base_idx: int, variant_id: int):
        if self.cache_npz_dir is None:
            return None
        name = f"sample_{base_idx:07d}_variant_{variant_id:03d}_seed_{self.seed}.npz"
        return self.cache_npz_dir / name

    @staticmethod
    def _atomic_save_npz(path: Path, defected_pos: np.ndarray) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, defected_pos=defected_pos)
        os.replace(tmp_path, path)

    def __len__(self):
        return len(self.dataset) * self.num_variants

    def __getitem__(self, idx):
        base_idx = idx // self.num_variants
        variant_id = idx % self.num_variants

        # Get the base data
        data = self.dataset[base_idx]

        # Validate input
        if not isinstance(data, Data) or not hasattr(data, "pos"):
            return None  # vadný vzorek

        # Input should already be normalized, so use pos directly
        pos = data.pos
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos).float()
        else:
            pos = pos.float()

        # Get the defect for this variant
        defect = self.defects[variant_id]

        # Seeding: use base_idx + variant_id + self.seed for consistent but different seeds
        sample_seed = self.seed + base_idx * self.num_variants + variant_id

        # Create proper random generators with deterministic seed
        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)

        # Store original for potential later use
        original_pos = pos.clone()

        cache_path = self._cache_path(base_idx=base_idx, variant_id=variant_id)
        defected_pos = None
        defect_log = {}

        if cache_path is not None and self.cache_read and cache_path.exists():
            with np.load(cache_path) as cached:
                defected_pos = np.asarray(cached["defected_pos"], dtype=np.float32)

        if defected_pos is None:
            # Convert to numpy for defect application
            pos_np = pos.numpy() if torch.is_tensor(pos) else np.asarray(pos)
            pos_np = pos_np.copy()  # Don't modify in-place

            # Apply defect with seeded randomness
            defected_pos, defect_log = defect.apply(pos_np)
            defected_pos = np.asarray(defected_pos, dtype=np.float32)

            if cache_path is not None and self.cache_write:
                self._atomic_save_npz(cache_path, defected_pos)

        # Convert back to tensor
        defected_pos_t = torch.from_numpy(defected_pos).float()

        # Build output Data object
        output = Data(
            original_pos=original_pos,
            defected_pos=defected_pos_t,
        )

        # Optionally include original and logs
        if self.detailed:
            output.category = getattr(data, "category", None)
            output.log = {defect.name: defect_log}
            if cache_path is not None:
                output.cache_path = str(cache_path)

        return output
