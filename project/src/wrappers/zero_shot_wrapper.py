import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points, knn_points
from visualize.utils import plot_pointcloud_to_image

class ZeroShotPatchWrapper(nn.Module):
    def __init__(self, model, patch_size=1024, num_patches=16, overlap_ratio=0.3):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.overlap_ratio = overlap_ratio
        self.novelty_threshold = 0.001

    def extract_patches(self, xyz: torch.Tensor):
        """
        xyz: (1, N, 3)
        Vrací:
            patches: (1, num_patches, patch_size, 3)
            centers: (1, num_patches, 3)
        """
        centers, _ = sample_farthest_points(xyz, K=self.num_patches)
        knn = knn_points(centers, xyz, K=self.patch_size, return_nn=True)
        return knn.knn, centers

    def _normalize_patch(self, patch: torch.Tensor):
        centroid = patch.mean(dim=1, keepdim=True)
        centered = patch - centroid
        scale = centered.norm(dim=2).max(dim=1)[0].clamp(min=1e-6).view(-1, 1, 1)
        return centered / scale, centroid, scale

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor, target_n: int = None) -> torch.Tensor:
        assert xyz.shape[0] == 1
        target_n = target_n or xyz.shape[1]

        patches, centers = self.extract_patches(xyz)

        all_new_points = []

        for i in range(self.num_patches):
            patch = patches[:, i, :, :]
            patch_norm, centroid, scale = self._normalize_patch(patch)

            _, fine = self.model(patch_norm)
            plot_pointcloud_to_image(patch_norm[0], f"patch_{i}.png")
            plot_pointcloud_to_image(fine[0], f"fine_{i}.png")
            fine_global = fine * scale + centroid  # (1, M, 3)

            # Zachovej jen body které jsou DÁLE od vstupních bodů
            # = model přidal něco nového, ne jen rekonstruoval existující body
            dists = knn_points(fine_global, patch, K=1).dists  # (1, M, 1)
            novelty_mask = (dists.squeeze(-1) > self.novelty_threshold).squeeze(0)
            
            new_pts = fine_global.squeeze(0)[novelty_mask]
            if new_pts.shape[0] > 0:
                all_new_points.append(new_pts)

        # Výstup = původní body + nové body od modelu
        original_pts = xyz.squeeze(0)  # (N, 3)
        
        if all_new_points:
            new_pts = torch.cat(all_new_points, dim=0)  # (K, 3)
            merged = torch.cat([original_pts, new_pts], dim=0).unsqueeze(0)  # (1, N+K, 3)
        else:
            merged = xyz

        output, _ = sample_farthest_points(merged, K=target_n)
        return output
