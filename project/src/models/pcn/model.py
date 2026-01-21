import torch
import torch.nn as nn

from .encoder import PointNetEncoder
from .decoder import PCNDecoder
from .loss import pcn_loss


class PCNRepairNet(nn.Module):
    def __init__(
        self,
        feat_dim: int = 1024,
        num_coarse: int = 1024,
        grid_size: int = 4,
        w_coarse: float = 0.5,
        w_fine: float = 1.0,
    ):
        super().__init__()
        self.encoder = PointNetEncoder(feat_dim)
        self.decoder = PCNDecoder(feat_dim, num_coarse, grid_size)
        self.w_coarse = w_coarse
        self.w_fine = w_fine

    def forward(self, bad: torch.Tensor):
        feat = self.encoder(bad)
        return self.decoder(feat)

    def compute_loss(self, pred, gt: torch.Tensor) -> torch.Tensor:
        coarse, fine = pred
        return pcn_loss(coarse, fine, gt, self.w_coarse, self.w_fine)
