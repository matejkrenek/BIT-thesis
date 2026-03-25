import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class NormalizeWrapperDataset(Dataset):
    def __init__(self, dataset, eps: float = 1e-12):
        """
        Args:
            dataset: Base dataset
            eps: Small constant to avoid division by zero
        """
        self.dataset = dataset
        self.eps = eps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        pos = data.pos
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos)
        pos = pos.float()

        # Center the point cloud
        centroid = pos.mean(dim=0, keepdim=True)
        pos_centered = pos - centroid

        # Scale to unit sphere
        scale = pos_centered.norm(dim=1).max().clamp(min=self.eps)
        pos_normalized = pos_centered / scale

        # Create output Data object
        output = Data(pos=pos_normalized, category=getattr(data, "category", None))

        return output
