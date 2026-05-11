"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: __init__.py
Responsibility: Exposes point cloud evaluation metrics and unified compute_metrics API for 3D shape completion/denoising tasks.
"""

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
    """
    Compute Chamfer, density-aware Chamfer, F-score and Hausdorff metrics for batched point clouds.

    Args:
        pred: (B, N, 3) or (N, 3) predicted point cloud(s).
        gt: (B, M, 3) or (M, 3) ground-truth point cloud(s).
        fscore_threshold: Distance threshold for F-score.
        pred_lengths: Optional tensor of valid point counts for pred.
        gt_lengths: Optional tensor of valid point counts for gt.
        density_alpha: Alpha parameter for density-aware Chamfer.

    Returns:
        Dictionary with keys: chamfer_distance, density_aware_chamfer_distance, fscore, hausdorff_distance.
    """
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
    # core metrics
    "chamfer_distance_metric",
    "density_aware_chamfer_distance_metric",
    "fscore_metric",
    "hausdorff_distance_metric",
    # unified API
    "compute_metrics",
]
