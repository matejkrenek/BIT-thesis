import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PatchWrapperDataset(Dataset):
    """
    Patch-based wrapper that extracts patches from point clouds.

    Each sample returns one or more patches extracted from the input point cloud.
    Patches are sampled stochastically on-the-fly for regularization.

    Supports multiple sampling strategies:
    - 'random': Uniformly random patch center + points within sphere
    - 'fps': Farthest Point Sampling for patch centers
    """

    def __init__(
        self,
        dataset: Dataset,
        patch_size: int = 512,
        num_patches: int = 1,
        sampling_strategy: str = "random",
        seed: int = 42,
    ):
        """
        Args:
            dataset: Base dataset (should provide Data with 'pos' attribute)
            patch_size: Number of points per patch
            num_patches: Number of patches to extract per sample
            sampling_strategy: 'random' or 'fps'
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.sampling_strategy = sampling_strategy
        self.seed = seed

        assert sampling_strategy in [
            "random",
            "fps",
        ], f"Unknown strategy: {sampling_strategy}"
        assert patch_size > 0, "patch_size must be > 0"
        assert num_patches > 0, "num_patches must be > 0"

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _farthest_point_sampling(pos: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Farthest Point Sampling using Euclidean distance.

        Args:
            pos: Point cloud [N, 3]
            num_samples: Number of points to sample

        Returns:
            Indices of sampled points [num_samples]
        """
        N = pos.shape[0]
        if num_samples >= N:
            return torch.arange(N)

        # Start with random point
        indices = torch.zeros(num_samples, dtype=torch.long)
        distances = torch.ones(N) * float("inf")

        # Random initialization
        indices[0] = torch.randint(0, N, (1,)).item()
        distances[indices[0]] = 0

        for i in range(1, num_samples):
            # Find farthest point
            idx = torch.argmax(distances)
            indices[i] = idx

            # Update distances
            point = pos[idx : idx + 1]  # [1, 3]
            dists = torch.norm(pos - point, dim=1)  # [N]
            distances = torch.min(distances, dists)

        return indices

    @staticmethod
    def _extract_patches_random(
        pos: torch.Tensor,
        num_patches: int,
        patch_size: int,
        radius: float = 0.5,
        rng: np.random.RandomState = None,
    ) -> list:
        """
        Extract patches by random center points.

        Args:
            pos: Point cloud [N, 3]
            num_patches: Number of patches
            patch_size: Points per patch
            radius: Radius around center point
            rng: Random number generator

        Returns:
            List of patch tensors [num_patches, patch_size, 3]
        """
        if rng is None:
            rng = np.random.RandomState()

        N = pos.shape[0]
        patches = []

        for _ in range(num_patches):
            # Random center point
            center_idx = rng.randint(0, N)
            center = pos[center_idx]

            # Find points within radius
            distances = torch.norm(pos - center, dim=1)
            mask = distances <= radius
            valid_indices = torch.where(mask)[0]

            # If not enough points in radius, use largest points within radius + closest outside
            if len(valid_indices) < patch_size:
                # Use all within radius + closest outside
                sorted_indices = torch.argsort(distances)
                valid_indices = sorted_indices[:patch_size]
            else:
                # Randomly sample from valid indices
                valid_indices = valid_indices[
                    torch.randperm(len(valid_indices))[:patch_size]
                ]

            patch = pos[valid_indices]  # [patch_size, 3]
            patches.append(patch)

        return patches

    @staticmethod
    def _extract_patches_fps(
        pos: torch.Tensor,
        num_patches: int,
        patch_size: int,
        radius: float = 0.5,
        rng: np.random.RandomState = None,
    ) -> list:
        """
        Extract patches using FPS for center selection.

        Args:
            pos: Point cloud [N, 3]
            num_patches: Number of patches
            patch_size: Points per patch
            radius: Radius around center point
            rng: Random number generator (unused, for API consistency)

        Returns:
            List of patch tensors [num_patches, patch_size, 3]
        """
        # Sample patch centers using FPS
        center_indices = PatchWrapperDataset._farthest_point_sampling(pos, num_patches)

        patches = []
        for center_idx in center_indices:
            center = pos[center_idx]

            # Find points within radius
            distances = torch.norm(pos - center, dim=1)
            mask = distances <= radius
            valid_indices = torch.where(mask)[0]

            # Ensure we have enough points
            if len(valid_indices) < patch_size:
                sorted_indices = torch.argsort(distances)
                valid_indices = sorted_indices[:patch_size]
            else:
                # Randomly sample from valid indices
                perm = torch.randperm(len(valid_indices))[:patch_size]
                valid_indices = valid_indices[perm]

            patch = pos[valid_indices]  # [patch_size, 3]
            patches.append(patch)

        return patches

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Validate input
        if not isinstance(data, Data) or not hasattr(data, "pos"):
            raise ValueError(f"Expected Data object with 'pos' at idx={idx}")

        pos = data.pos
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos).float()
        else:
            pos = pos.float()

        # Create seeded RNG
        sample_seed = self.seed + idx
        rng = np.random.RandomState(sample_seed)

        # Extract patches based on strategy
        if self.sampling_strategy == "random":
            patches = self._extract_patches_random(
                pos, self.num_patches, self.patch_size, radius=0.5, rng=rng
            )
        else:  # fps
            patches = self._extract_patches_fps(
                pos, self.num_patches, self.patch_size, radius=0.5, rng=rng
            )

        # Stack patches or return single patch
        if self.num_patches == 1:
            patches_tensor = patches[0]  # [patch_size, 3]
        else:
            patches_tensor = torch.stack(patches)  # [num_patches, patch_size, 3]

        # Create output Data object
        output = Data(
            pos=patches_tensor,
            category=getattr(data, "category", None),
        )

        # Store patch metadata
        output.num_patches = self.num_patches
        output.patch_size = self.patch_size

        return output
