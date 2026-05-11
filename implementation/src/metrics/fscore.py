"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: fscore.py
Responsibility: F-score metric for 3D point clouds using distance thresholding.
"""

from typing import Optional

import torch
from pytorch3d.ops import knn_points

from .types import Reduction
from .utils import ensure_batched, masked_mean, reduce_values


def fscore_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
    pred_lengths: Optional[torch.Tensor] = None,
    gt_lengths: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute symmetric F-score with distance threshold in Euclidean space.

    Args:
        pred: (B, N, 3) or (N, 3) predicted point cloud(s).
        gt: (B, M, 3) or (M, 3) ground-truth point cloud(s).
        threshold: Distance threshold for match.
        pred_lengths: Optional tensor of valid point counts for pred.
        gt_lengths: Optional tensor of valid point counts for gt.
        reduction: Reduction over batch ('mean', 'sum', 'none').
        eps: Small value to avoid division by zero.

    Returns:
        F-score as a scalar tensor or batch tensor.
    """
    pred = ensure_batched(pred)
    gt = ensure_batched(gt)

    d_pred_to_gt = torch.sqrt(
        knn_points(
            pred, gt, K=1, lengths1=pred_lengths, lengths2=gt_lengths
        ).dists.squeeze(-1)
    )
    d_gt_to_pred = torch.sqrt(
        knn_points(
            gt, pred, K=1, lengths1=gt_lengths, lengths2=pred_lengths
        ).dists.squeeze(-1)
    )

    precision = masked_mean((d_pred_to_gt <= threshold).float(), pred_lengths)
    recall = masked_mean((d_gt_to_pred <= threshold).float(), gt_lengths)
    values = (2.0 * precision * recall) / (precision + recall + eps)
    return reduce_values(values, reduction)


__all__ = ["fscore_metric"]
