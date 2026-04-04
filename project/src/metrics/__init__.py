"""Reusable point-cloud evaluation metrics package."""

from typing import Dict, Optional

import torch

from .chamfer import chamfer_distance_metric
from .density_aware_chamfer import density_aware_chamfer_distance_metric
from .fscore import fscore_metric
from .hausdorff import hausdorff_distance_metric


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    fscore_threshold: float,
    pred_lengths: Optional[torch.Tensor] = None,
    gt_lengths: Optional[torch.Tensor] = None,
    density_alpha: float = 1000.0,
) -> Dict[str, float]:
    """Compute Chamfer, density-aware Chamfer, F-score and Hausdorff metrics."""
    cd = chamfer_distance_metric(
        pred, gt, pred_lengths=pred_lengths, gt_lengths=gt_lengths
    )
    dcd = density_aware_chamfer_distance_metric(
        pred,
        gt,
        pred_lengths=pred_lengths,
        gt_lengths=gt_lengths,
        alpha=density_alpha,
    )
    f1 = fscore_metric(
        pred,
        gt,
        threshold=fscore_threshold,
        pred_lengths=pred_lengths,
        gt_lengths=gt_lengths,
    )
    hd = hausdorff_distance_metric(
        pred, gt, pred_lengths=pred_lengths, gt_lengths=gt_lengths
    )

    return {
        "chamfer_distance": cd.item(),
        "density_aware_chamfer_distance": dcd.item(),
        "fscore": f1.item(),
        "hausdorff_distance": hd.item(),
    }


__all__ = [
    "chamfer_distance_metric",
    "density_aware_chamfer_distance_metric",
    "fscore_metric",
    "hausdorff_distance_metric",
    "compute_metrics",
]
