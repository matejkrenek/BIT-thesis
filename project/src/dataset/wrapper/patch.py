import math
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PatchWrapperDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        patch_size: int = 8192,
        num_patches: int | None = None,
        normalize_patches: bool = False,
        overlap_ratio: float = 0.5,
        max_extra_patches: int | None = None,
    ):
        self.dataset = dataset
        self.patch_size = int(patch_size)
        self.num_patches = num_patches
        self.normalize_patches = normalize_patches
        self.overlap_ratio = float(overlap_ratio)
        self.max_extra_patches = max_extra_patches

        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0.0, 1.0)")

    def __len__(self):
        return len(self.dataset)

    def _extract_pos_and_meta(self, sample):
        if isinstance(sample, Data):
            # Preferred format from AugmentWrapperDataset.
            if (
                hasattr(sample, "original_pos")
                and sample.original_pos is not None
                and hasattr(sample, "defected_pos")
                and sample.defected_pos is not None
            ):
                return sample.original_pos, sample.defected_pos, sample

            # Backward-compatible fallback for datasets with single point cloud in `pos`.
            if hasattr(sample, "pos") and sample.pos is not None:
                return sample.pos, sample.pos, sample

            raise ValueError(
                "Wrapped Data sample must contain `original_pos` and `defected_pos` "
                "(or fallback `pos`)."
            )
        if isinstance(sample, dict):
            if "original_pos" in sample and "defected_pos" in sample:
                return sample["original_pos"], sample["defected_pos"], sample
            if "pos" in sample:
                return sample["pos"], sample["pos"], sample
            raise ValueError(
                "Wrapped dict sample must contain keys `original_pos` and `defected_pos` "
                "(or fallback `pos`)."
            )

        raise TypeError(
            f"Unsupported sample type: {type(sample)}. "
            "Expected torch_geometric.data.Data or dict sample."
        )

    @staticmethod
    def _knn_indices(
        points: torch.Tensor, centers: torch.Tensor, k: int
    ) -> torch.Tensor:
        """
        points: (N, 3), centers: (M, 3)
        returns: (M, k)
        """
        n = points.shape[0]
        dists = torch.cdist(centers, points)

        if n >= k:
            return torch.topk(dists, k=k, largest=False, dim=1).indices

        nearest = torch.topk(dists, k=n, largest=False, dim=1).indices
        rep = math.ceil(k / n)
        return nearest.repeat(1, rep)[:, :k]

    @staticmethod
    def _normalize_patch(patch: torch.Tensor):
        centroid = patch.mean(dim=0, keepdim=True)
        centered = patch - centroid
        scale = centered.norm(dim=1).max().clamp_min(1e-8)
        normalized = centered / scale
        return normalized, centroid.squeeze(0), scale

    def _build_fps_knn_patches(
        self, pos: torch.Tensor, base_patch_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = pos.shape[0]
        effective_unique_per_patch = max(
            int(self.patch_size * (1.0 - self.overlap_ratio)), 1
        )
        overlap_patch_count = math.ceil(n / effective_unique_per_patch)

        target_count = max(base_patch_count, overlap_patch_count)
        target_count = min(target_count, n)

        centers, _ = sample_farthest_points(pos.unsqueeze(0), K=target_count)
        centers = centers.squeeze(0)

        patch_indices = self._knn_indices(pos, centers, self.patch_size)
        covered = torch.zeros(n, dtype=torch.bool, device=pos.device)
        covered[patch_indices.reshape(-1).unique()] = True

        uncovered = torch.arange(n, device=pos.device)[~covered]
        if uncovered.numel() == 0:
            return patch_indices, centers

        if self.max_extra_patches is None:
            max_extra = uncovered.numel()
        else:
            max_extra = max(int(self.max_extra_patches), 0)

        extra_patches = []
        extra_centers = []
        current_uncovered = uncovered
        while current_uncovered.numel() > 0 and len(extra_patches) < max_extra:
            extra_center = pos[current_uncovered[0]].unsqueeze(0)
            extra_idx = self._knn_indices(pos, extra_center, self.patch_size).squeeze(0)
            extra_patches.append(extra_idx)
            extra_centers.append(extra_center.squeeze(0))

            covered[extra_idx.unique()] = True
            current_uncovered = torch.arange(n, device=pos.device)[~covered]

        if len(extra_patches) > 0:
            patch_indices = torch.cat(
                [patch_indices, torch.stack(extra_patches, dim=0)], dim=0
            )
            centers = torch.cat([centers, torch.stack(extra_centers, dim=0)], dim=0)

        return patch_indices, centers

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        original_pos, defected_pos, data = self._extract_pos_and_meta(sample)

        if not torch.is_tensor(original_pos):
            original_pos = torch.tensor(original_pos, dtype=torch.float32)
        if not torch.is_tensor(defected_pos):
            defected_pos = torch.tensor(defected_pos, dtype=torch.float32)

        original_pos = original_pos.float()
        defected_pos = defected_pos.float()

        if original_pos.ndim != 2 or original_pos.shape[1] != 3:
            raise ValueError(
                f"`original_pos` must have shape (N, 3), got {tuple(original_pos.shape)}"
            )
        if defected_pos.ndim != 2 or defected_pos.shape[1] != 3:
            raise ValueError(
                f"`defected_pos` must have shape (N, 3), got {tuple(defected_pos.shape)}"
            )

        n_points_defected = defected_pos.shape[0]
        if n_points_defected == 0:
            raise ValueError("Defected point cloud is empty.")

        if original_pos.shape[0] == 0:
            raise ValueError("Original point cloud is empty.")

        m = (
            self.num_patches
            if self.num_patches is not None
            else math.ceil(n_points_defected / self.patch_size)
        )
        m = max(1, int(m))

        # Build patch centers and indices from defected cloud, then gather paired
        # patches from both defected and original cloud using the same centers.
        defected_patch_indices, centers = self._build_fps_knn_patches(defected_pos, m)
        original_patch_indices = self._knn_indices(
            original_pos, centers, self.patch_size
        )

        defected_patches = defected_pos[defected_patch_indices]  # (M, K, 3)
        original_patches = original_pos[original_patch_indices]  # (M, K, 3)

        if self.normalize_patches:
            original_norm = [self._normalize_patch(p)[0] for p in original_patches]
            defected_norm = [self._normalize_patch(p)[0] for p in defected_patches]
            original_patches_tensor = torch.stack(original_norm, dim=0)
            defected_patches_tensor = torch.stack(defected_norm, dim=0)
        else:
            original_patches_tensor = original_patches
            defected_patches_tensor = defected_patches

        output = Data(
            original_pos=original_patches_tensor,
            defected_pos=defected_patches_tensor,
            # Keep `pos` for compatibility with code that expects a single tensor.
            pos=defected_patches_tensor,
            category=(
                getattr(data, "category", None)
                if isinstance(data, Data)
                else data.get("category", None)
            ),
        )
        output.num_patches = int(defected_patches_tensor.shape[0])
        output.patch_size = self.patch_size
        output.overlap_ratio = self.overlap_ratio

        covered_points = defected_patch_indices.reshape(-1).unique().numel()
        output.coverage_ratio = float(covered_points / n_points_defected)
        return output
