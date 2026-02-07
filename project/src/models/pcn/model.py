import torch
import torch.nn as nn
from .encoder import PCNEncoder
from .decoder import PCNDecoder
from pytorch3d.loss import chamfer_distance


class PCN(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()

        self.encoder = PCNEncoder(latent_dim)
        self.decoder = PCNDecoder(num_dense, latent_dim, grid_size)

    def forward(self, xyz):
        latent = self.encoder(xyz)
        coarse, fine = self.decoder(latent)
        return coarse, fine

    def repulsion_loss(self, points, k=5, h=0.03):
        """
        points: (B, N, 3)
        k: number of neighbors
        h: minimal desired distance
        """
        B, N, _ = points.shape

        # pairwise distances
        dist = torch.cdist(points, points)  # (B, N, N)

        # ignore self-distance
        dist += torch.eye(N, device=points.device).unsqueeze(0) * 1e6

        # k nearest neighbors
        knn_dist, _ = torch.topk(dist, k=k, largest=False)

        # penalize distances smaller than h
        loss = torch.clamp(h - knn_dist, min=0.0)
        return loss.mean()

    def compute_loss(self, pred, target):
        coarse, fine = pred

        loss_coarse, _ = chamfer_distance(coarse, target)
        loss_fine, _ = chamfer_distance(fine, target)

        return loss_coarse + 0.25 * loss_fine

    def improved_pcn_loss(self, pred_coarse, pred_fine, gt, w_c=0.3, w_f=1.0, w_r=0.05):
        loss_c, _ = chamfer_distance(pred_coarse, gt)
        loss_f, _ = chamfer_distance(pred_fine, gt)
        loss_r = self.repulsion_loss(pred_fine)

        return w_c * loss_c + w_f * loss_f + w_r * loss_r
