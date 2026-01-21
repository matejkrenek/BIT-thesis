import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int = 1024,
        num_coarse: int = 1024,
        grid_size: int = 4,
    ):
        super().__init__()

        self.num_coarse = num_coarse
        self.grid_size = grid_size
        self.num_fine = num_coarse * (grid_size**2)

        self.fc1 = nn.Linear(feat_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.fold1 = nn.Linear(feat_dim + 3 + 2, 512)
        self.fold2 = nn.Linear(512, 512)
        self.fold3 = nn.Linear(512, 3)

    def forward(self, feat: torch.Tensor):
        B = feat.shape[0]

        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        coarse = self.fc3(x).view(B, self.num_coarse, 3)

        lin = torch.linspace(-0.05, 0.05, self.grid_size, device=feat.device)
        gx, gy = torch.meshgrid(lin, lin, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)

        S = self.grid_size**2
        grid = grid.repeat(B, self.num_coarse, 1, 1).view(B, -1, 2)

        coarse_rep = coarse.unsqueeze(2).repeat(1, 1, S, 1).view(B, -1, 3)
        feat_rep = feat.unsqueeze(1).repeat(1, coarse_rep.shape[1], 1)

        fold_in = torch.cat([feat_rep, coarse_rep, grid], dim=-1)
        y = F.relu(self.fold1(fold_in))
        y = F.relu(self.fold2(y))
        delta = self.fold3(y)

        fine = coarse_rep + delta
        return coarse, fine
