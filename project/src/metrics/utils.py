from typing import Optional

import torch

from .types import Reduction


def ensure_batched(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 2:
        return points.unsqueeze(0)
    if points.ndim == 3:
        return points
    raise ValueError(f"Expected shape (N, 3) or (B, N, 3), got {tuple(points.shape)}")


def validity_mask(lengths: torch.Tensor, max_points: int, device: torch.device) -> torch.Tensor:
    ids = torch.arange(max_points, device=device).unsqueeze(0)
    return ids < lengths.unsqueeze(1)


def masked_mean(values: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    if lengths is None:
        return values.mean(dim=1)

    mask = validity_mask(lengths, values.shape[1], values.device)
    denom = lengths.clamp(min=1).to(values.dtype)
    return (values * mask).sum(dim=1) / denom


def masked_max(values: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    if lengths is None:
        return values.max(dim=1).values

    mask = validity_mask(lengths, values.shape[1], values.device)
    masked = values.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values


def reduce_values(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
