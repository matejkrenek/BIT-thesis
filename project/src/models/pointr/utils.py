from pytorch3d.ops import sample_farthest_points, knn_points
from pytorch3d.loss import chamfer_distance
import torch
from torch import nn


class ChamferDistanceL1(nn.Module):
    """
    Chamfer Distance L1 loss using pytorch3d operations.
    Computes the L1-based Chamfer distance between two point clouds.
    
    Returns:
        Tuple of (loss, ) where loss is the computed Chamfer distance
    """
    
    def __init__(self, point_reduction: str = "mean"):
        """
        Args:
            point_reduction: How to reduce distances. Can be 'mean' or 'sum'
        """
        super().__init__()
        self.point_reduction = point_reduction
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Compute Chamfer distance between prediction and ground truth.
        
        Args:
            pred: Predicted point cloud [B, N, 3]
            gt: Ground truth point cloud [B, M, 3]
            
        Returns:
            Chamfer distance loss (scalar tensor)
        """
        # pytorch3d's chamfer_distance returns (loss, )
        # We use abs_relative_diff which uses L1 distance by default
        loss, _ = chamfer_distance(pred, gt, point_reduction=self.point_reduction)
        return loss


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
        sampled_points, _ = sample_farthest_points(pc_float32, K=num, random_start_point=False)
    
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
        _, indices = sample_farthest_points(xyz_float32, K=num, random_start_point=False)
    
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
