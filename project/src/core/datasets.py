from typing import Callable, Optional, Tuple

import numpy as np
from torch.utils.data.dataset import Dataset

from dataset import ShapeNetDataset
from dataset.defect import (
    Combined,
    LargeMissingRegion,
    LocalDropout,
    Noise,
    OutlierPoints,
    SurfaceToPlaneBridge,
    BelowObjectPlane,
)
from dataset.wrapper import (
    AugmentWrapperDataset,
    DenseWrapperDataset,
    NormalizeWrapperDataset,
    PatchWrapperDataset,
    StagedAugmentWrapperDataset,
)
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset


def _prepare_dataset_pipeline(
    base_dataset: Dataset,
    defects: list,
    dense: bool,
    dense_root: Optional[str],
    dense_num_points: int,
    normalize: bool,
    split_into_patches: bool,
    patch_size: int,
    num_patches: Optional[int],
    normalize_patches: bool,
    overlap_ratio: float,
    max_extra_patches: Optional[int],
    patching_method: str,
    patch_radius: float,
    patch_center: str,
    patch_point_count_std: float,
    include_full_objects_in_patches: bool,
    seed: int,
) -> Dataset:
    dataset: Dataset = base_dataset

    if dense:
        if not dense_root:
            raise ValueError("dense_root must be provided when dense=True")
        dataset = DenseWrapperDataset(
            dataset=dataset,
            root=dense_root,
            num_points=dense_num_points,
        )

    if normalize:
        dataset = NormalizeWrapperDataset(dataset)

    dataset = AugmentWrapperDataset(
        dataset=dataset,
        defects=defects,
        seed=seed,
    )

    if split_into_patches:
        dataset = PatchWrapperDataset(
            dataset=dataset,
            patch_size=patch_size,
            num_patches=num_patches,
            normalize_patches=normalize_patches,
            overlap_ratio=overlap_ratio,
            max_extra_patches=max_extra_patches,
            patching_method=patching_method,
            patch_radius=patch_radius,
            patch_center=patch_center,
            patch_point_count_std=patch_point_count_std,
            include_full_objects=include_full_objects_in_patches,
        )

    return dataset


def create_basic_reconstruction_dataset(
    base_dataset: Optional[Dataset] = None,
    root: Optional[str] = None,
    seed: int = 42,
    defect_augmentation_count: int = 5,
    local_dropout_regions: int = 5,
    dense: bool = False,
    dense_root: Optional[str] = None,
    dense_num_points: int = 100_000,
    normalize: bool = True,
    visualize: bool = False,
    split_into_patches: bool = False,
    patch_size: int = 8192,
    num_patches: Optional[int] = None,
    normalize_patches: bool = False,
    overlap_ratio: float = 0.5,
    max_extra_patches: Optional[int] = None,
    patching_method: str = "fps_knn",
    patch_radius: float = 0.05,
    patch_center: str = "point",
    patch_point_count_std: float = 0.0,
    include_full_objects_in_patches: bool = False,
) -> Dataset:
    """
    Build a reconstruction dataset with structural missing parts and small local holes.

    Defects:
      - LargeMissingRegion (missing parts)
      - LocalDropout (small holes)
    """
    if base_dataset is None:
        if not root:
            raise ValueError("Either base_dataset or root must be provided")
        base_dataset = ShapeNetDataset(root=root)

    rng = np.random.RandomState(seed)
    defects = [
        Combined(
            [
                LargeMissingRegion(removal_fraction=rng.uniform(0.1, 0.35)),
                LocalDropout(
                    radius=rng.uniform(0.01, 0.06),
                    regions=local_dropout_regions,
                    dropout_rate=rng.uniform(0.45, 0.9),
                ),
            ]
        )
        for _ in range(defect_augmentation_count)
    ]

    dataset = _prepare_dataset_pipeline(
        base_dataset=base_dataset,
        defects=defects,
        dense=dense,
        dense_root=dense_root,
        dense_num_points=dense_num_points,
        normalize=normalize,
        split_into_patches=split_into_patches,
        patch_size=patch_size,
        num_patches=num_patches,
        normalize_patches=normalize_patches,
        overlap_ratio=overlap_ratio,
        max_extra_patches=max_extra_patches,
        patching_method=patching_method,
        patch_radius=patch_radius,
        patch_center=patch_center,
        patch_point_count_std=patch_point_count_std,
        include_full_objects_in_patches=include_full_objects_in_patches,
        seed=seed,
    )

    if visualize:
        from visualize.viewer import SampleViewer

        viewer = SampleViewer(
            dataset,
            inference=None,
        )

        viewer.show()

    return dataset


def create_advanced_reconstruction_dataset(
    base_dataset: Optional[Dataset] = None,
    root: Optional[str] = None,
    seed: int = 42,
    defect_augmentation_count: int = 5,
    local_dropout_regions: int = 5,
    dense: bool = False,
    dense_root: Optional[str] = None,
    dense_num_points: int = 100_000,
    normalize: bool = True,
    visualize: bool = False,
    split_into_patches: bool = False,
    patch_size: int = 8192,
    num_patches: Optional[int] = None,
    normalize_patches: bool = False,
    overlap_ratio: float = 0.5,
    max_extra_patches: Optional[int] = None,
    patching_method: str = "fps_knn",
    patch_radius: float = 0.05,
    patch_center: str = "point",
    patch_point_count_std: float = 0.0,
    include_full_objects_in_patches: bool = False,
) -> Dataset:
    """
    Build advanced reconstruction in two stages:
    1) Create the basic reconstruction dataset (holes / missing parts).
    2) Add only noise + outliers on top of the basic defected cloud.

    Output semantics per sample:
      - original_pos: basic defected cloud (with holes)
      - defected_pos: basic defected cloud + noise/outliers
    """
    if base_dataset is None:
        if not root:
            raise ValueError("Either base_dataset or root must be provided")
        base_dataset = ShapeNetDataset(root=root)

    rng = np.random.RandomState(seed)
    stage2_defects = [
        Combined(
            [
                SurfaceToPlaneBridge(
                    num_bridges=int(rng.randint(6, 18)),
                    points_per_bridge=int(rng.randint(8, 24)),
                    plane_offset_ratio=rng.uniform(0.02, 0.08),
                    axis=1,
                    bottom_band_ratio=rng.uniform(0.2, 0.5),
                    top_band_ratio=rng.uniform(0.15, 0.35),
                    side_band_ratio=rng.uniform(0.15, 0.35),
                    bottom_bridge_fraction=rng.uniform(0.1, 0.35),
                    top_bridge_fraction=rng.uniform(0.2, 0.4),
                    side_bridge_fraction=rng.uniform(0.35, 0.6),
                    diagonal_strength_min=rng.uniform(0.1, 0.25),
                    diagonal_strength_max=rng.uniform(0.4, 0.75),
                    lateral_jitter=rng.uniform(0.0008, 0.0035),
                    normal_jitter=rng.uniform(0.0005, 0.0025),
                ),
                BelowObjectPlane(
                    num_points=int(rng.randint(500, 2200)),
                    offset_ratio=rng.uniform(0.02, 0.08),
                    spread_ratio=rng.uniform(1.0, 2.0),
                    normal_jitter=rng.uniform(0.0008, 0.004),
                    plane_jitter=rng.uniform(0.003, 0.02),
                    axis=1,
                    center_density_bias=rng.uniform(0.25, 0.75),
                    edge_sparsity=rng.uniform(0.2, 0.75),
                    boundary_irregularity=rng.uniform(0.2, 0.7),
                ),
                OutlierPoints(
                    num_points=int(rng.randint(30, 240)),
                    scale_factor=rng.uniform(1.2, 2.0),
                    mode="uniform",
                ),
                Noise(sigma=rng.uniform(0.001, 0.01)),
            ]
        )
        for _ in range(defect_augmentation_count)
    ]

    basic_dataset = create_basic_reconstruction_dataset(
        base_dataset=base_dataset,
        root=None,
        seed=seed,
        defect_augmentation_count=defect_augmentation_count,
        local_dropout_regions=local_dropout_regions,
        dense=dense,
        dense_root=dense_root,
        dense_num_points=dense_num_points,
        normalize=normalize,
        visualize=False,
        split_into_patches=False,
        patch_size=patch_size,
        num_patches=num_patches,
        normalize_patches=normalize_patches,
        overlap_ratio=overlap_ratio,
        max_extra_patches=max_extra_patches,
        patching_method=patching_method,
        patch_radius=patch_radius,
        patch_center=patch_center,
        patch_point_count_std=patch_point_count_std,
        include_full_objects_in_patches=False,
    )

    dataset: Dataset = StagedAugmentWrapperDataset(
        dataset=basic_dataset,
        defects=stage2_defects,
        seed=seed,
    )

    if split_into_patches:
        dataset = PatchWrapperDataset(
            dataset=dataset,
            patch_size=patch_size,
            num_patches=num_patches,
            normalize_patches=normalize_patches,
            overlap_ratio=overlap_ratio,
            max_extra_patches=max_extra_patches,
            patching_method=patching_method,
            patch_radius=patch_radius,
            patch_center=patch_center,
            patch_point_count_std=patch_point_count_std,
            include_full_objects=include_full_objects_in_patches,
        )

    if visualize:
        from visualize.viewer import SampleViewer

        viewer = SampleViewer(
            dataset,
            inference=None,
        )

        viewer.show()

    return dataset


def _collate_fn(batch):
    def _extract_pair(item):
        if item is None:
            return None

        if isinstance(item, dict):
            if "original_pos" in item and "defected_pos" in item:
                return item["original_pos"], item["defected_pos"]
            if "pos" in item:
                return item["pos"], item["pos"]

        if hasattr(item, "original_pos") and hasattr(item, "defected_pos"):
            return item.original_pos, item.defected_pos
        if hasattr(item, "pos"):
            return item.pos, item.pos

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[0], item[1]

        raise TypeError(
            f"Unsupported batch item type for collate: {type(item).__name__}"
        )

    pairs = []
    for item in batch:
        if item is None:
            continue
        pair = _extract_pair(item)
        if pair is None:
            continue
        pairs.append(pair)

    if len(pairs) == 0:
        return None, None, None

    originals_list = []
    defecteds_list = []
    for original, defected in pairs:
        if not torch.is_tensor(original):
            original = torch.as_tensor(original)
        if not torch.is_tensor(defected):
            defected = torch.as_tensor(defected)

        originals_list.append(original.float())
        defecteds_list.append(defected.float())

    originals = torch.stack(originals_list, dim=0)
    lengths = torch.tensor([pc.shape[0] for pc in defecteds_list], dtype=torch.long)
    max_n = int(lengths.max().item())

    padded = torch.zeros(len(defecteds_list), max_n, 3, dtype=originals.dtype)
    for idx, pc in enumerate(defecteds_list):
        padded[idx, : pc.shape[0]] = pc

    return originals, padded, lengths


def create_train_val_test_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    collate_fn: Optional[Callable] = _collate_fn,
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
        "generator": torch_generator,
    }

    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_kwargs)

    return train_loader, val_loader, test_loader
