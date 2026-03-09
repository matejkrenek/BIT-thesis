"""Reusable point-cloud evaluation metrics package."""

from typing import Dict, Optional

import torch

from .chamfer import chamfer_distance_metric
from .fscore import fscore_metric
from .hausdorff import hausdorff_distance_metric


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    fscore_threshold: float,
    pred_lengths: Optional[torch.Tensor] = None,
    gt_lengths: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute Chamfer Distance, F-score and Hausdorff Distance."""
    cd = chamfer_distance_metric(pred, gt, pred_lengths=pred_lengths, gt_lengths=gt_lengths)
    f1 = fscore_metric(
        pred,
        gt,
        threshold=fscore_threshold,
        pred_lengths=pred_lengths,
        gt_lengths=gt_lengths,
    )
    hd = hausdorff_distance_metric(pred, gt, pred_lengths=pred_lengths, gt_lengths=gt_lengths)

    return {
        "chamfer_distance": cd.item(),
        "fscore": f1.item(),
        "hausdorff_distance": hd.item(),
    }


__all__ = [
    "chamfer_distance_metric",
    "fscore_metric",
    "hausdorff_distance_metric",
    "compute_metrics",
]
