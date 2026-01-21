import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, feat_dim: int = 1024):
        super().__init__()
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)  # (B, N, F)
        x = torch.max(x, dim=1).values  # (B, F)
        return x
