"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: split.py
Responsibility: Wrapper dataset for splitting datasets into train/val/test subsets or arbitrary partitions.
"""

import numpy as np
from torch.utils.data import Dataset


class SplitWrapperDataset(Dataset):
    """
    Wrapper dataset for splitting a dataset into train/val/test subsets or arbitrary partitions.

    Args:
        dataset (Dataset): Base dataset to split.
        split (str): Which split to use ('train', 'val', 'test').
        split_ratio (tuple): Fractions for train/val/test (must sum to 1.0).
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        assert split in [
            "train",
            "val",
            "test",
        ], "split must be 'train', 'val', or 'test'"
        self.dataset = dataset
        rng = np.random.RandomState(seed)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        n = len(indices)
        train_end = int(split_ratio[0] * n)
        val_end = train_end + int(split_ratio[1] * n)
        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Get a sample from the split by index."""
        return self.dataset[self.indices[idx]]
