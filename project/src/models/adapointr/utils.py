from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points, knn_points
import torch
from torch import nn


def _filter_ignore_zeros(xyz1: torch.Tensor, xyz2: torch.Tensor, ignore_zeros: bool):
    """Match official behavior: only filter zeros when batch size is 1."""
    if xyz1.size(0) == 1 and ignore_zeros:
        non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
        non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
        xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
        xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
    return xyz1, xyz2


class ChamferDistanceL1(nn.Module):
    """Official-style Chamfer L1: (mean(sqrt(dist1)) + mean(sqrt(dist2))) / 2."""

    def __init__(self, ignore_zeros: bool = False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
        xyz1, xyz2 = _filter_ignore_zeros(xyz1, xyz2, self.ignore_zeros)
        # Use PyTorch3D chamfer to obtain per-point squared-L2 directional terms.
        (dist1, dist2), _ = chamfer_distance(
            xyz1,
            xyz2,
            batch_reduction=None,
            point_reduction=None,
            norm=2,
            single_directional=False,
        )
        dist1 = torch.sqrt(dist1.clamp_min(1e-12))
        dist2 = torch.sqrt(dist2.clamp_min(1e-12))
        # Match official averaging: mean over points then mean over batch.
        return ((dist1.mean(dim=1) + dist2.mean(dim=1)) / 2).mean()


def fps(pc: torch.Tensor, num: int) -> torch.Tensor:
    """
    Farthest Point Sampling using pytorch3d operations.

    Args:
        pc: Point cloud tensor of shape [B, N, 3] or [B, 3, N]
        num: Number of points to sample

    Returns:
        Sampled point cloud of shape [B, num, 3]
    """
    # Handle input format: if [B, 3, N], transpose to [B, N, 3]
    if pc.shape[1] == 3 and pc.shape[2] != 3:
        pc = pc.transpose(1, 2).contiguous()

    pc = pc.contiguous()

    # Apply farthest point sampling using pytorch3d
    try:
        sampled_points, _ = sample_farthest_points(pc, K=num, random_start_point=False)
    except RuntimeError:
        # Fallback with float32 conversion if the above fails
        pc_float32 = pc.float()
        sampled_points, _ = sample_farthest_points(
            pc_float32, K=num, random_start_point=False
        )

    return sampled_points


def fps_indices(xyz: torch.Tensor, num: int) -> torch.Tensor:
    """
    Farthest Point Sampling indices using pytorch3d operations.

    Args:
        xyz: Point cloud tensor of shape [B, N, 3] - must be on the same device
        num: Number of points to sample

    Returns:
        Indices of sampled points of shape [B, num]
    """
    # Ensure tensor is contiguous and on the correct device
    xyz = xyz.contiguous()

    # Use sample_farthest_points without random_start_point for stability
    # This is more reliable for variable-length point clouds
    try:
        _, indices = sample_farthest_points(xyz, K=num, random_start_point=False)
    except RuntimeError:
        # Fallback: use a simpler dtype conversion if the above fails
        xyz_float32 = xyz.float()
        _, indices = sample_farthest_points(
            xyz_float32, K=num, random_start_point=False
        )

    return indices


def gather_operation(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather features using indices (equivalent to pointnet2_utils.gather_operation).

    Args:
        features: Feature tensor of shape [B, C, N]
        indices: Indices tensor of shape [B, num_group]

    Returns:
        Gathered features of shape [B, C, num_group]
    """
    batch_size, num_channels, _ = features.shape
    _, num_group = indices.shape

    if indices.dtype != torch.long:
        indices = indices.long()

    if indices.numel() > 0:
        min_idx = int(indices.min().item())
        max_idx = int(indices.max().item())
        max_valid = features.shape[2] - 1
        if min_idx < 0 or max_idx > max_valid:
            raise RuntimeError(
                "gather_operation index out of bounds: "
                f"min={min_idx}, max={max_idx}, valid=[0,{max_valid}]"
            )

    # Expand indices to match feature dimensions for gathering
    indices_expanded = indices.unsqueeze(1).expand(batch_size, num_channels, num_group)

    # Gather along the last dimension
    gathered = torch.gather(features, 2, indices_expanded)

    return gathered


def fps_sample(xyz: torch.Tensor, m: int) -> torch.Tensor:
    """
    Farthest Point Sampling using pytorch3d operations.

    Args:
        xyz: [B, N, 3]
        m: Number of points to sample

    Returns:
        centers: [B, m, 3]
    """
    xyz = xyz.contiguous()
    try:
        centers, _ = sample_farthest_points(xyz, K=m, random_start_point=False)
    except RuntimeError:
        # Fallback with float32 conversion
        xyz_float32 = xyz.float()
        centers, _ = sample_farthest_points(xyz_float32, K=m, random_start_point=False)
    return centers


def gather_knn(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz: [B, N, 3]
    centers: [B, M, 3]
    returns neigh: [B, M, k, 3]
    """
    knn = knn_points(centers, xyz, K=k, return_nn=True)
    return knn.knn  # [B, M, k, 3]


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance, output shape [B, N, M]."""
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src**2, dim=-1, keepdim=True)
    dist += torch.sum(dst**2, dim=-1).unsqueeze(1)
    return dist


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """KNN indices with shape [B, S, nsample] for queries in new_xyz over xyz."""
    knn = knn_points(new_xyz.contiguous(), xyz.contiguous(), K=nsample)
    return knn.idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points/features by index on dim=1, supports [B, S] and [B, S, K]."""
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(batch_size, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def three_nn(
    unknown: torch.Tensor, known: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 3-NN distances and indices for unknown points against known points."""
    knn = knn_points(unknown.contiguous(), known.contiguous(), K=3)
    return torch.sqrt(knn.dists.clamp_min(1e-12)), knn.idx


def three_interpolate(
    features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Interpolate features with 3-NN weights.

    Args:
        features: [B, C, N]
        idx: [B, M, 3]
        weight: [B, M, 3]
    Returns:
        [B, C, M]
    """
    features_t = features.transpose(1, 2).contiguous()  # [B, N, C]
    gathered = index_points(features_t, idx)  # [B, M, 3, C]
    weighted = gathered * weight.unsqueeze(-1)
    interpolated = weighted.sum(dim=2)  # [B, M, C]
    return interpolated.transpose(1, 2).contiguous()  # [B, C, M]


def jitter_points(
    points: torch.Tensor,
    sigma: float = 0.01,
    clip: float = 0.05,
) -> torch.Tensor:
    """Apply bounded Gaussian jitter to point coordinates."""
    noise = torch.randn_like(points) * sigma
    noise = noise.clamp(min=-clip, max=clip)
    return points + noise
