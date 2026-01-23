import torch
import torch.nn as nn


class PCNDecoder(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()

        assert num_dense % (grid_size**2) == 0

        self.num_dense = num_dense
        self.grid_size = grid_size
        self.num_coarse = num_dense // (grid_size**2)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse),
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        # folding seed (registered as buffer → správně se přesouvá mezi CPU/GPU)
        a = torch.linspace(-0.05, 0.05, steps=grid_size).view(1, grid_size)
        b = torch.linspace(-0.05, 0.05, steps=grid_size).view(grid_size, 1)

        grid = torch.cat(
            [
                a.expand(grid_size, grid_size).reshape(1, -1),
                b.expand(grid_size, grid_size).reshape(1, -1),
            ],
            dim=0,
        ).view(1, 2, grid_size**2)

        self.register_buffer("folding_seed", grid)

    def forward(self, latent):
        B = latent.shape[0]

        coarse = self.mlp(latent).view(B, self.num_coarse, 3)  # (B, Nc, 3)

        point_feat = (
            coarse.unsqueeze(2)
            .expand(-1, -1, self.grid_size**2, -1)
            .reshape(B, self.num_dense, 3)
            .transpose(2, 1)
        )  # (B, 3, Nf)

        seed = (
            self.folding_seed.unsqueeze(2)
            .expand(B, -1, self.num_coarse, -1)
            .reshape(B, 2, self.num_dense)
        )

        latent_feat = latent.unsqueeze(2).expand(-1, -1, self.num_dense)

        feat = torch.cat([latent_feat, seed, point_feat], dim=1)
        fine = self.final_conv(feat) + point_feat

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()
