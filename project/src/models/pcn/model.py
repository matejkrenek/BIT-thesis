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

    def compute_loss(self, pred, target):
        coarse, fine = pred

        loss_coarse, _ = chamfer_distance(coarse, target)
        loss_fine, _ = chamfer_distance(fine, target)

        return loss_coarse + loss_fine
