import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import LargeMissingRegion, Rotate, Noise, LocalDropout, Combined
from models import PCN
from pytorch3d.ops import sample_farthest_points
from pytorch3d.loss import chamfer_distance
import random as rnd
from visualize.utils import plot_pointcloud_to_image
from dataset.utils import create_augmented_dataset, create_split_dataloaders
from models import PoinTr
from wrappers.zero_shot_wrapper import ZeroShotPatchWrapper
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pytorch3d.ops import sample_farthest_points, knn_points

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"
CHECKPOINT_PATH = DATA_FOLDER_PATH + "/checkpoints/pointr/v1_best.pt"

BATCH_SIZE = 1
NUM_WORKERS = 4

SEED = 42
g = torch.Generator()
g.manual_seed(SEED)
rng = np.random.RandomState(SEED)
np.random.seed(SEED)

# -----------------------
# Dataset
# -----------------------
base_dataset = ShapeNetDataset(
    root=ROOT_DATA,
)
augmented_dataset = create_augmented_dataset(
    base=base_dataset,
    defect_augmentation_count=5,
    local_dropout_regions=5,
    seed=SEED,
)
train_loader, val_loader, test_loader = create_split_dataloaders(
    dataset=augmented_dataset,
    batch_size=BATCH_SIZE,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=SEED,
    num_workers=NUM_WORKERS,
)


# -----------------------
# Model
# -----------------------
class PoinTrConfig:
    trans_dim = 384
    knn_layer = 1
    num_pred = 16384
    num_query = 224


model = PoinTr(config=PoinTrConfig()).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"[INFO] Loaded checkpoint: {CHECKPOINT_PATH}")
print(f"[INFO] Evaluating on {len(test_loader)} samples")

# -----------------------
# Evaluation
# -----------------------

from visualize.viewer import SampleViewer


def model_inference(model, xyz):
    with torch.no_grad():
        _, pred = model(xyz)

    return pred.squeeze(0).cpu()


visualizer = SampleViewer(
    dataset=train_loader,
    inference=lambda sample: model_inference(model, sample[1].unsqueeze(0).to(DEVICE)),
)

visualizer.show()
