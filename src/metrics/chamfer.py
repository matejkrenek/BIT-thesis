"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: chamfer.py
Responsibility: Chamfer distance metric for 3D point clouds using PyTorch3D.
"""

from typing import Optional

import torch
from pytorch3d.loss import chamfer_distance

from .types import Reduction
from .utils import ensure_batched


def chamfer_distance_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    pred_lengths: Optional[torch.Tensor] = None,
    gt_lengths: Optional[torch.Tensor] = None,
    batch_reduction: Optional[Reduction] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
) -> torch.Tensor:
    """
    Compute Chamfer Distance using PyTorch3D.

    Args:
        pred: (B, N, 3) or (N, 3) predicted point cloud(s).
        gt: (B, M, 3) or (M, 3) ground-truth point cloud(s).
        pred_lengths: Optional tensor of valid point counts for pred.
        gt_lengths: Optional tensor of valid point counts for gt.
        batch_reduction: Reduction over batch ('mean', 'sum', 'none').
        point_reduction: Reduction over points ('mean', 'sum', 'none').
        norm: Distance norm (default 2).

    Returns:
        Chamfer distance as a scalar tensor or batch tensor.
    """
    pred = ensure_batched(pred)
    gt = ensure_batched(gt)

    if batch_reduction == "none":
        batch_reduction = None

    cd, _ = chamfer_distance(
        pred,
        gt,
        x_lengths=pred_lengths,
        y_lengths=gt_lengths,
        batch_reduction=batch_reduction,
        point_reduction=point_reduction,
        norm=norm,
    )
    return cd


__all__ = ["chamfer_distance_metric"]
