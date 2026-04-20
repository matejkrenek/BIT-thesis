from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from pytorch3d.ops import knn_points

from models.pointr.utils import ChamferDistanceL1, fps


class PointMAEPatchEncoder(nn.Module):
    """Point-MAE style patch encoder for grouped neighborhoods."""

    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = int(encoder_channel)
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
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        # point_groups: [B, G, S, 3]
        bs, groups, samples, _ = point_groups.shape
        x = point_groups.reshape(bs * groups, samples, 3)
        x = self.first_conv(x.transpose(2, 1))
        x_global = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.cat([x_global.expand(-1, -1, samples), x], dim=1)
        x = self.second_conv(x)
        x = torch.max(x, dim=2, keepdim=False)[0]
        return x.reshape(bs, groups, self.encoder_channel)


class PointMAEGrouping(nn.Module):
    """FPS + kNN grouping with local normalization around patch centers."""

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz: [B, N, 3]
        center = fps(xyz, self.num_group)
        knn = knn_points(center, xyz, K=self.group_size, return_nn=True)
        neighborhood = knn.knn
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood.contiguous(), center.contiguous()


class Fold(nn.Module):
    def __init__(self, in_channel: int, step: int, hidden_dim: int = 512):
        super().__init__()
        self.in_channel = int(in_channel)
        self.step = int(step)

        a = (
            torch.linspace(-1.0, 1.0, steps=step, dtype=torch.float)
            .view(1, step)
            .expand(step, step)
            .reshape(1, -1)
        )
        b = (
            torch.linspace(-1.0, 1.0, steps=step, dtype=torch.float)
            .view(step, 1)
            .expand(step, step)
            .reshape(1, -1)
        )
        self.register_buffer("folding_seed", torch.cat([a, b], dim=0))

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(
            bs, self.in_channel, num_sample
        )
        seed = (
            self.folding_seed.view(1, 2, num_sample)
            .expand(bs, 2, num_sample)
            .to(x.device)
        )

        y = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(y)
        y = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(y)
        return fd2


class PointMAECompletionBackbone(nn.Module):
    """Point-MAE inspired encoder + transformer memory for completion queries."""

    def __init__(
        self,
        *,
        num_group: int,
        group_size: int,
        trans_dim: int,
        encoder_dims: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        decoder_depth: int,
        num_query: int,
    ):
        super().__init__()
        self.num_query = int(num_query)

        self.group_divider = PointMAEGrouping(
            num_group=num_group, group_size=group_size
        )
        self.patch_encoder = PointMAEPatchEncoder(encoder_channel=encoder_dims)
        self.token_proj = (
            nn.Linear(encoder_dims, trans_dim)
            if int(encoder_dims) != int(trans_dim)
            else nn.Identity()
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=num_heads,
            dim_feedforward=max(int(trans_dim * mlp_ratio), trans_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(trans_dim)

        self.global_proj = nn.Sequential(
            nn.Conv1d(trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
        )

        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * self.num_query),
        )

        self.query_mlp = nn.Sequential(
            nn.Linear(1024 + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, trans_dim),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=trans_dim,
            nhead=num_heads,
            dim_feedforward=max(int(trans_dim * mlp_ratio), trans_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(trans_dim)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = xyz.size(0)
        neighborhood, centers = self.group_divider(xyz)
        tokens = self.patch_encoder(neighborhood)
        tokens = self.token_proj(tokens)
        memory = self.encoder(tokens + self.pos_embed(centers))
        memory = self.encoder_norm(memory)

        global_feature = self.global_proj(memory.transpose(1, 2))
        global_feature = torch.max(global_feature, dim=-1, keepdim=False)[0]

        coarse = self.coarse_pred(global_feature).reshape(bs, self.num_query, 3)
        query_tokens = self.query_mlp(
            torch.cat(
                [global_feature.unsqueeze(1).expand(-1, self.num_query, -1), coarse],
                dim=-1,
            )
        )

        q = self.decoder(tgt=query_tokens, memory=memory)
        q = self.decoder_norm(q)
        return q, coarse


class PointMAECompletion(nn.Module):
    """Completion model that uses Point-MAE style encoding on incomplete point clouds."""

    def __init__(
        self,
        trans_dim: int = 384,
        num_pred: int = 16384,
        num_query: int = 224,
        num_group: int = 128,
        group_size: int = 32,
        encoder_dims: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        decoder_depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pointmae_ckpt: str | None = None,
    ):
        super().__init__()
        self.trans_dim = int(trans_dim)
        self.num_pred = int(num_pred)
        self.num_query = int(num_query)

        step_f = (self.num_pred / max(self.num_query, 1)) ** 0.5
        self.fold_step = int(step_f + 0.5)
        if self.fold_step * self.fold_step * self.num_query != self.num_pred:
            raise ValueError(
                "num_pred must equal num_query * k^2 for folding decoder. "
                f"Got num_pred={self.num_pred}, num_query={self.num_query}."
            )

        self.backbone = PointMAECompletionBackbone(
            num_group=num_group,
            group_size=group_size,
            trans_dim=self.trans_dim,
            encoder_dims=encoder_dims,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            decoder_depth=decoder_depth,
            num_query=self.num_query,
        )

        self.foldingnet = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.loss_func = ChamferDistanceL1()

        if pointmae_ckpt:
            self.load_pointmae_encoder(pointmae_ckpt)

    def load_pointmae_encoder(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load compatible Point-MAE encoder weights (partial load is expected)."""
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("base_model", checkpoint.get("model", checkpoint))

        if not isinstance(state, Mapping):
            raise TypeError(
                f"Checkpoint at {path} does not contain a valid state dict mapping."
            )

        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            k = str(key)
            if k.startswith("module."):
                k = k[len("module.") :]

            if k.startswith("MAE_encoder.encoder."):
                mapped = "backbone.patch_encoder." + k[len("MAE_encoder.encoder.") :]
                cleaned[mapped] = value
            elif k.startswith("MAE_encoder.pos_embed."):
                mapped = "backbone.pos_embed." + k[len("MAE_encoder.pos_embed.") :]
                cleaned[mapped] = value

        incompatible = self.load_state_dict(cleaned, strict=False)
        return {
            "loaded_keys": len(cleaned),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    def get_loss(self, ret, gt, epoch: int = 0):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz: torch.Tensor):
        q, coarse_point_cloud = self.backbone(xyz)

        bsz, num_tokens, _ = q.shape
        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)
        global_feature = torch.max(global_feature, dim=1, keepdim=False)[0]

        rebuild_feature = torch.cat(
            [
                global_feature.unsqueeze(-2).expand(-1, num_tokens, -1),
                q,
                coarse_point_cloud,
            ],
            dim=-1,
        )

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(bsz * num_tokens, -1))
        relative_xyz = self.foldingnet(rebuild_feature).reshape(bsz, num_tokens, 3, -1)
        rebuild_points = (
            (relative_xyz + coarse_point_cloud.unsqueeze(-1))
            .transpose(2, 3)
            .reshape(bsz, -1, 3)
        )

        inp_sparse = fps(xyz, self.num_query)
        coarse_point_cloud = torch.cat(
            [coarse_point_cloud, inp_sparse], dim=1
        ).contiguous()
        rebuild_points = torch.cat([rebuild_points, xyz], dim=1).contiguous()

        return coarse_point_cloud, rebuild_points
