import torch
from pytorch3d.loss import chamfer_distance


def pcn_loss(
    pred_coarse: torch.Tensor,
    pred_fine: torch.Tensor,
    gt: torch.Tensor,
    w_coarse: float = 0.5,
    w_fine: float = 1.0,
) -> torch.Tensor:
    loss_c, _ = chamfer_distance(pred_coarse, gt)
    loss_f, _ = chamfer_distance(pred_fine, gt)
    return w_coarse * loss_c + w_fine * loss_f
