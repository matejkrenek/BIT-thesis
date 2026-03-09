from typing import Optional

import torch
from pytorch3d.ops import knn_points

from .types import Reduction
from .utils import ensure_batched, masked_max, reduce_values


def hausdorff_distance_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    pred_lengths: Optional[torch.Tensor] = None,
    gt_lengths: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Compute symmetric Hausdorff distance in Euclidean space."""
    pred = ensure_batched(pred)
    gt = ensure_batched(gt)

    d_pred_to_gt = torch.sqrt(
        knn_points(pred, gt, K=1, lengths1=pred_lengths, lengths2=gt_lengths).dists.squeeze(-1)
    )
    d_gt_to_pred = torch.sqrt(
        knn_points(gt, pred, K=1, lengths1=gt_lengths, lengths2=pred_lengths).dists.squeeze(-1)
    )

    max_pred_to_gt = masked_max(d_pred_to_gt, pred_lengths)
    max_gt_to_pred = masked_max(d_gt_to_pred, gt_lengths)
    values = torch.maximum(max_pred_to_gt, max_gt_to_pred)
    return reduce_values(values, reduction)
