"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: loss.py
Responsibility: Loss helper utilities for PCN coarse-to-fine reconstruction supervision.

Attribution:
    This module is adapted from the official PoinTr PCN implementation:
    https://github.com/yuxumin/PoinTr/blob/master/models/PCN.py
    The adaptation exposes reusable loss computation for local training pipelines.
"""

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
