"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: utils.py
Responsibility: Utility functions for batched point cloud metrics (masking, reduction, shape checks).
"""

from typing import Optional

import torch

from .types import Reduction


def ensure_batched(points: torch.Tensor) -> torch.Tensor:
    """
    Ensure input is batched: (N, 3) -> (1, N, 3), (B, N, 3) unchanged.
    """
    if points.ndim == 2:
        return points.unsqueeze(0)
    if points.ndim == 3:
        return points
    raise ValueError(f"Expected shape (N, 3) or (B, N, 3), got {tuple(points.shape)}")


def validity_mask(
    lengths: torch.Tensor, max_points: int, device: torch.device
) -> torch.Tensor:
    """
    Create boolean mask for valid points in padded batched clouds.
    """
    ids = torch.arange(max_points, device=device).unsqueeze(0)
    return ids < lengths.unsqueeze(1)


def masked_mean(values: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compute mean over valid (unpadded) points for each batch.
    """
    if lengths is None:
        return values.mean(dim=1)
    mask = validity_mask(lengths, values.shape[1], values.device)
    denom = lengths.clamp(min=1).to(values.dtype)
    return (values * mask).sum(dim=1) / denom


def masked_max(values: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compute max over valid (unpadded) points for each batch.
    """
    if lengths is None:
        return values.max(dim=1).values
    mask = validity_mask(lengths, values.shape[1], values.device)
    masked = values.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values


def reduce_values(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    """
    Reduce tensor by 'mean', 'sum', or return as is ('none').
    """
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


__all__ = [
    "ensure_batched",
    "validity_mask",
    "masked_mean",
    "masked_max",
    "reduce_values",
]
