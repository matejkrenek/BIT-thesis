import torch
import torch.nn as nn


class PCNEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, latent_dim, 1),
        )

    def forward(self, xyz):
        # xyz: (B, N, 3)
        x = self.first_conv(xyz.transpose(2, 1))  # (B, 256, N)

        x_global = torch.max(x, dim=2, keepdim=True)[0]  # (B, 256, 1)
        x = torch.cat([x_global.expand_as(x), x], dim=1)  # (B, 512, N)

        x = self.second_conv(x)  # (B, latent, N)
        latent = torch.max(x, dim=2)[0]  # (B, latent)

        return latent
