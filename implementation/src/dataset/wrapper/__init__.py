"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: __init__.py
Responsibility: Exports dataset wrapper classes for normalization, patching, augmentation, splitting, and compatibility.
"""

from .dense import DenseWrapperDataset
from .augment import AugmentWrapperDataset
from .patch import PatchWrapperDataset
from .normalize import NormalizeWrapperDataset
from .staged_augment import StagedAugmentWrapperDataset
from .pointcleannet_compat import (
    PointcloudPatchDataset,
)

__all__ = [
    "DenseWrapperDataset",
    "AugmentWrapperDataset",
    "PatchWrapperDataset",
    "NormalizeWrapperDataset",
    "StagedAugmentWrapperDataset",
    "PointcloudPatchDataset",
]
