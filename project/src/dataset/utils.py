from dataset import AugmentedDataset, ShapeNetDataset
from dataset.defect import Combined, LargeMissingRegion, LocalDropout, Noise
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple


def create_augmented_dataset(
    base: Dataset,
    seed: int,
    defect_augmentation_count: int = 5,
    local_dropout_regions: int = 5,
) -> AugmentedDataset:
    """
    Creates an augmented dataset by applying random transformations to the original dataset.

    Args:
        dataset (ShapeNetDataset): The original ShapeNet dataset to augment.
        num_augmentations (int): The number of augmented samples to create for each original sample.
        noise_std (float): The standard deviation of the Gaussian noise to add to the point clouds.

    Returns:
        AugmentedDataset: A new dataset containing the augmented samples.
    """
    rng = np.random.RandomState(seed)
    defects = [
        Combined(
            [
                LargeMissingRegion(removal_fraction=rng.uniform(0.1, 0.3)),
                LocalDropout(
                    radius=rng.uniform(0.01, 0.1),
                    regions=local_dropout_regions,
                    dropout_rate=rng.uniform(0.5, 0.9),
                ),
            ]
        )
        for _ in range(defect_augmentation_count)
    ]
    return AugmentedDataset(dataset=base, defects=defects)


def create_split_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split a dataset into train/val/test subsets and return data loaders.

    The test ratio is inferred as ``1 - train_ratio - val_ratio``.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    dataset_len = len(dataset)
    train_size = int(dataset_len * train_ratio)
    val_size = int(dataset_len * val_ratio)
    test_size = dataset_len - train_size - val_size

    torch_generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch_generator,
    )

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }

    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_kwargs)

    return train_loader, val_loader, test_loader
