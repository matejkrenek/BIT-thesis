import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class StagedAugmentWrapperDataset(Dataset):
    """Apply a second-stage defect on top of an already defected sample.

    Input sample is expected to contain:
    - original_pos: clean/base reference
    - defected_pos: first-stage defected cloud (e.g., with holes)

    Output sample semantics:
    - original_pos: first-stage defected cloud
    - defected_pos: first-stage cloud + second-stage artifacts
    """

    def __init__(
        self,
        dataset: Dataset,
        defects: list,
        detailed: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.defects = defects
        self.detailed = detailed
        self.seed = int(seed)

        if len(defects) == 0:
            raise ValueError("At least one second-stage defect must be provided")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if not isinstance(data, Data):
            return None
        if not hasattr(data, "original_pos") or not hasattr(data, "defected_pos"):
            return None

        base_original = torch.as_tensor(data.original_pos).float()
        base_defected = torch.as_tensor(data.defected_pos).float()

        defect = self.defects[int(idx) % len(self.defects)]
        sample_seed = self.seed + int(idx)

        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)

        defected_np = base_defected.detach().cpu().numpy().copy()
        second_stage_pos, defect_log = defect.apply(defected_np)
        second_stage_pos_t = torch.from_numpy(second_stage_pos).float()

        output = Data(
            original_pos=base_defected.clone(),
            defected_pos=second_stage_pos_t,
            category=getattr(data, "category", None),
        )

        if self.detailed:
            output.log = {
                "stage1": "basic_reconstruction",
                "stage2": {defect.name: defect_log},
            }
            output.base_original_pos = base_original

        return output
