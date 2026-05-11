"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: model.py
Responsibility: PCN model wrapper that composes encoder/decoder and exposes training losses.

Attribution:
    This module is adapted from the official PoinTr PCN implementation:
    https://github.com/yuxumin/PoinTr/blob/master/models/PCN.py
    The adaptation keeps PCN behavior while splitting the model into local encoder,
    decoder, and helper-loss modules.
"""

import torch
import torch.nn as nn
from .encoder import PCNEncoder
from .decoder import PCNDecoder
from pytorch3d.loss import chamfer_distance


class PCN(nn.Module):
    """Point Completion Network with coarse-to-fine decoding."""

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()

        self.encoder = PCNEncoder(latent_dim)
        self.decoder = PCNDecoder(num_dense, latent_dim, grid_size)

    def forward(self, xyz):
        """Predict coarse and fine completed point clouds from partial input."""
        latent = self.encoder(xyz)
        coarse, fine = self.decoder(latent)
        return coarse, fine

    def repulsion_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Memory-safe repulsion loss using random subsampling.
        points: (B, N, 3)
        """
        B, N, _ = points.shape
        S = min(1024, N)

        # random subsample (same indices for whole batch)
        idx = torch.randperm(N, device=points.device)[:S]
        pts = points[:, idx, :]  # (B, S, 3)

        # pairwise distances on subsample
        dist = torch.cdist(pts, pts)  # (B, S, S)

        # ignore self-distance
        dist += torch.eye(S, device=points.device).unsqueeze(0) * 1e6

        # k nearest neighbors
        knn_dist, _ = torch.topk(dist, k=5, largest=False)

        # penalize points that are too close
        loss = torch.clamp(0.03 - knn_dist, min=0.0)
        return loss.mean()

    def compute_loss(self, pred, target, w_coarse=0.1, w_fine=1):
        """Compute weighted Chamfer loss for coarse and fine outputs."""
        coarse, fine = pred

        loss_coarse, _ = chamfer_distance(coarse, target)
        loss_fine, _ = chamfer_distance(fine, target)

        return loss_coarse * w_coarse + loss_fine * w_fine

    def improved_pcn_loss(self, pred, gt, w_c=0.3, w_f=1.0, w_r=0.05):
        """Compute Chamfer losses with an additional repulsion regularizer."""
        pred_coarse, pred_fine = pred

        loss_c, _ = chamfer_distance(pred_coarse, gt)
        loss_f, _ = chamfer_distance(pred_fine, gt)
        loss_r = self.repulsion_loss(pred_fine)

        return w_c * loss_c + w_f * loss_f + w_r * loss_r
