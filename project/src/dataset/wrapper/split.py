import numpy as np
from torch.utils.data import Dataset


class SplitWrapperDataset(Dataset):
    def __init__(
        self,
        dataset,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        seed=42,
    ):
        assert split in ["train", "val", "test"]

        self.dataset = dataset

        rng = np.random.RandomState(seed)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

        n = len(indices)
        train_end = int(split_ratio[0] * n)
        val_end = train_end + int(split_ratio[1] * n)

        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
