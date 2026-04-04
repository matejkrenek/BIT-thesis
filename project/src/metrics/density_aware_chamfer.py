from typing import Optional

import torch
from pytorch3d.ops import knn_points

from .types import Reduction
from .utils import ensure_batched, masked_mean, reduce_values


def _directional_density_term(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_lengths: Optional[torch.Tensor],
    dst_lengths: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    knn = knn_points(src, dst, K=1, lengths1=src_lengths, lengths2=dst_lengths)
    sq_dists = knn.dists.squeeze(-1)
    nn_idx = knn.idx.squeeze(-1).long()

    batch_size, src_max_points = nn_idx.shape
    dst_max_points = dst.shape[1]

    device = src.device
    dtype = src.dtype

    if src_lengths is None:
        valid_src = torch.ones((batch_size, src_max_points), device=device, dtype=dtype)
    else:
        src_ids = torch.arange(src_max_points, device=device).unsqueeze(0)
        valid_src = (src_ids < src_lengths.unsqueeze(1)).to(dtype)

    counts = torch.zeros((batch_size, dst_max_points), device=device, dtype=dtype)
    counts.scatter_add_(1, nn_idx, valid_src)

    assigned_counts = counts.gather(1, nn_idx).clamp_min(1.0)
    transformed_dist = 1.0 - torch.exp(-alpha * sq_dists)
    weighted_terms = transformed_dist / assigned_counts

    return masked_mean(weighted_terms, src_lengths)


def density_aware_chamfer_distance_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    pred_lengths: Optional[torch.Tensor] = None,
    gt_lengths: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
    alpha: float = 1000.0,
) -> torch.Tensor:
    """Compute density-aware symmetric Chamfer distance.

    This variant discounts matches that collapse many source points onto the
    same nearest-neighbor target point, which improves sensitivity to local
    density mismatch compared to standard Chamfer distance.
    """
    pred = ensure_batched(pred)
    gt = ensure_batched(gt)

    pred_to_gt = _directional_density_term(
        pred,
        gt,
        src_lengths=pred_lengths,
        dst_lengths=gt_lengths,
        alpha=alpha,
    )
    gt_to_pred = _directional_density_term(
        gt,
        pred,
        src_lengths=gt_lengths,
        dst_lengths=pred_lengths,
        alpha=alpha,
    )

    values = 0.5 * (pred_to_gt + gt_to_pred)
    return reduce_values(values, reduction)
