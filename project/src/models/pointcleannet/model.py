import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


class ResidualMLP1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class STNkd(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, k * k),
        )

        # Start close to identity transform for stability.
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        feat = self.conv(x)
        global_feat = torch.max(feat, dim=2)[0]
        transform = self.fc(global_feat)
        identity = (
            torch.eye(self.k, device=x.device, dtype=x.dtype)
            .view(1, self.k * self.k)
            .repeat(batch_size, 1)
        )
        transform = transform + identity
        return transform.view(batch_size, self.k, self.k)


class PointFeatureEncoder(nn.Module):
    def __init__(self, use_point_stn: bool = True, use_feat_stn: bool = True):
        super().__init__()
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn

        if use_point_stn:
            self.stn_input = STNkd(k=3)
        if use_feat_stn:
            self.stn_feat = STNkd(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = xyz.transpose(1, 2).contiguous()

        if self.use_point_stn:
            trans_in = self.stn_input(x)
            x = torch.bmm(trans_in, x)

        x = self.act(self.bn1(self.conv1(x)))
        point_feat = self.act(self.bn2(self.conv2(x)))

        if self.use_feat_stn:
            trans_feat = self.stn_feat(point_feat)
            point_feat = torch.bmm(trans_feat, point_feat)

        x = self.act(self.bn3(self.conv3(point_feat)))
        x = self.act(self.bn4(self.conv4(x)))

        global_feat = torch.max(x, dim=2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, xyz.shape[1])
        return point_feat, global_feat


class RefinementStage(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, max_offset: float):
        super().__init__()
        self.max_offset = max_offset
        self.block1 = ResidualMLP1D(in_channels, hidden_channels)
        self.block2 = ResidualMLP1D(hidden_channels, hidden_channels)
        self.block3 = ResidualMLP1D(hidden_channels, hidden_channels // 2)
        self.head = nn.Conv1d(hidden_channels // 2, 4, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.block1(feat)
        x = self.block2(x)
        x = self.block3(x)
        pred = self.head(x)
        offset = torch.tanh(pred[:, :3, :]) * self.max_offset
        confidence = torch.sigmoid(pred[:, 3:4, :])
        return offset * confidence


class PointCleanNetDenoiser(nn.Module):
    """
    PointCleanNet-inspired denoising network operating on full point clouds.

    Input:
        xyz: (B, N, 3)
    Output:
        denoised: (B, N, 3)

    The model predicts per-point displacement vectors from local neighborhoods
    and optional global context, then applies them to noisy coordinates.
    """

    def __init__(
        self,
        k_neighbors: int = 32,
        local_feature_dim: int = 128,
        hidden_dim: int = 256,
        num_stages: int = 2,
        max_offset: float = 0.05,
        query_chunk_size: int = 1024,
        use_point_stn: bool = True,
        use_feat_stn: bool = True,
    ):
        super().__init__()

        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")
        if local_feature_dim <= 0:
            raise ValueError("local_feature_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_stages <= 0:
            raise ValueError("num_stages must be > 0")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be > 0")
        if max_offset <= 0:
            raise ValueError("max_offset must be > 0")

        self.k_neighbors = k_neighbors
        self.max_offset = max_offset
        self.query_chunk_size = query_chunk_size
        self.num_stages = num_stages

        self.local_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, local_feature_dim, 1),
            nn.BatchNorm2d(local_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.point_encoder = PointFeatureEncoder(
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
        )

        fusion_in_dim = local_feature_dim + 64 + 256 + 3

        self.stages = nn.ModuleList(
            [
                RefinementStage(
                    in_channels=fusion_in_dim,
                    hidden_channels=hidden_dim,
                    max_offset=max_offset / max(1, num_stages),
                )
                for _ in range(num_stages)
            ]
        )

    def _effective_k(self, n_points: int) -> int:
        return max(1, min(self.k_neighbors, n_points))

    def _encode_local_patches(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Build per-point local features from KNN neighborhoods.

        Returns:
            local_features: (B, N, C_local)
        """
        _, N, _ = xyz.shape
        chunk_size = max(1, self.query_chunk_size)
        k = self._effective_k(N)

        encoded_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            query = xyz[:, start:end, :]  # (B, Q, 3)

            knn = knn_points(query, xyz, K=k, return_nn=True)
            neighbors = knn.knn  # (B, Q, K, 3)

            relative = neighbors - query.unsqueeze(2)
            pair_feat = torch.cat([relative, neighbors], dim=-1)
            pair_feat = pair_feat.permute(0, 3, 1, 2).contiguous()

            local_encoded = self.local_encoder(pair_feat)
            local_encoded = torch.max(local_encoded, dim=3)[0]
            encoded_chunks.append(local_encoded)

        local_features = torch.cat(encoded_chunks, dim=2).transpose(1, 2).contiguous()
        return local_features

    def forward(
        self,
        xyz: torch.Tensor,
        return_stages: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            xyz: noisy input point cloud, shape (B, N, 3)

        Returns:
            denoised point cloud, shape (B, N, 3)
        """
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"Expected input shape (B, N, 3), got {tuple(xyz.shape)}")
        if xyz.shape[1] < 2:
            raise ValueError("Input point cloud must contain at least 2 points")

        denoised = xyz
        stage_outputs: List[torch.Tensor] = []
        for stage in self.stages:
            local_features = self._encode_local_patches(denoised)
            point_feat, global_feat = self.point_encoder(denoised)

            fused = torch.cat(
                [
                    local_features,
                    point_feat.transpose(1, 2).contiguous(),
                    global_feat.transpose(1, 2).contiguous(),
                    denoised,
                ],
                dim=-1,
            )
            offset = stage(fused.transpose(1, 2).contiguous())
            denoised = denoised + offset.transpose(1, 2).contiguous()
            if return_stages:
                stage_outputs.append(denoised)

        if return_stages:
            return denoised, stage_outputs
        return denoised

    def _smoothness_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Local smoothness regularizer encouraging nearby denoised points to stay coherent.
        """
        k_smooth = min(points.shape[1], self.k_neighbors + 1)
        knn = knn_points(points, points, K=k_smooth, return_nn=True)
        neighbors = knn.knn[:, :, 1:, :] if k_smooth > 1 else knn.knn
        center = points.unsqueeze(2)
        return ((neighbors - center) ** 2).mean()

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_points: Optional[torch.Tensor] = None,
        w_chamfer: float = 1.0,
        w_consistency: float = 0.05,
        w_smooth: float = 0.0,
    ) -> torch.Tensor:
        """
        Denoising loss with Chamfer supervision and optional regularization.
        """
        loss_chamfer, _ = chamfer_distance(pred, target)
        loss = w_chamfer * loss_chamfer

        if input_points is not None and w_consistency > 0.0:
            loss_consistency = ((pred - input_points) ** 2).mean()
            loss = loss + w_consistency * loss_consistency

        if w_smooth > 0.0:
            loss_smooth = self._smoothness_loss(pred)
            loss = loss + w_smooth * loss_smooth

        return loss

    def get_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.compute_loss(pred=pred, target=target, input_points=input_points)
