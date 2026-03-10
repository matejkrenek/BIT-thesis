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
    """Compute Chamfer Distance using PyTorch3D."""
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
