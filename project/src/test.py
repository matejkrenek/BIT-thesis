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

test_dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root=ROOT_DATA),
    defects=[
        Combined(
            [
                LargeMissingRegion(removal_fraction=rng.uniform(0.1, 0.3)),
                LocalDropout(
                    radius=rng.uniform(0.01, 0.1),
                    regions=5,
                    dropout_rate=rng.uniform(0.5, 0.9),
                ),
                Noise(rng.uniform(0.001, 0.005)),
            ]
        )
        for _ in range(5)
    ],
)


def collate(batch):
    originals, defecteds = zip(*batch)

    originals = torch.stack(originals, dim=0)

    lengths = torch.tensor([pc.shape[0] for pc in defecteds], dtype=torch.long)
    max_n = lengths.max().item()

    padded = torch.zeros(len(defecteds), max_n, 3)
    for i, pc in enumerate(defecteds):
        padded[i, : pc.shape[0]] = pc

    return originals, padded, lengths


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate,
    pin_memory=True,
    generator=g,
)

# -----------------------
# Model
# -----------------------

class PoinTrConfig:
    trans_dim = 384
    knn_layer = 1
    num_pred = 16384
    num_query = 224

from models import PoinTr

model = PoinTr(config=PoinTrConfig()).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"[INFO] Loaded checkpoint: {CHECKPOINT_PATH}")
print(f"[INFO] Evaluating on {len(test_dataset)} samples")

# -----------------------
# Evaluation
# -----------------------

from wrappers.zero_shot_wrapper import ZeroShotPatchWrapper
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pytorch3d.ops import sample_farthest_points, knn_points

def _set_ax(ax, pts, elev, azim):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

def visualize_patches(
    xyz: torch.Tensor,
    wrapper: ZeroShotPatchWrapper,
    elev: float = 20,
    azim: float = 45,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    output_path: str = None,
):
    """
    xyz:     (1, N, 3)
    wrapper: ZeroShotPatchWrapper — použije jeho extract_patches
    """
    if isinstance(xyz, np.ndarray):
        xyz = torch.from_numpy(xyz).float()
    if xyz.ndim == 2:
        xyz = xyz.unsqueeze(0)
    xyz = xyz.cpu()

    with torch.no_grad():
        patches, centers = wrapper.extract_patches(xyz)  # (1, P, K, 3), (1, P, 3)

    pts   = xyz[0].numpy()       # (N, 3)
    ctrs  = centers[0].numpy()   # (P, 3)
    ptchs = patches[0].numpy()   # (P, K, 3)

    num_patches = wrapper.num_patches
    colors = cm.tab20(np.linspace(0, 1, num_patches))

    fig = plt.figure(figsize=figsize, dpi=dpi)

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                s=0.5, c="lightgray", linewidths=0, alpha=0.5)
    ax1.scatter(ctrs[:, 0], ctrs[:, 1], ctrs[:, 2],
                s=30, c=colors, linewidths=0.5, edgecolors="black", zorder=5)
    ax1.set_title(f"FPS středy (P={num_patches})", fontsize=9)
    _set_ax(ax1, pts, elev, azim)

    ax2 = fig.add_subplot(122, projection="3d")
    for i in range(num_patches):
        ax2.scatter(ptchs[i, :, 0], ptchs[i, :, 1], ptchs[i, :, 2],
                    s=0.5, color=colors[i], linewidths=0, alpha=0.7)
    ax2.scatter(ctrs[:, 0], ctrs[:, 1], ctrs[:, 2],
                s=30, c=colors, linewidths=0.5, edgecolors="black", zorder=5)
    ax2.set_title(f"KNN patche (K={wrapper.patch_size})", fontsize=9)
    _set_ax(ax2, pts, elev, azim)

    fig.tight_layout(pad=1.0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    return image

@torch.no_grad()
def evaluate():
    total_cd = 0.0

    wrapper = ZeroShotPatchWrapper(
        model=model,
        patch_size=8124,   # větší patch = více kontextu
        num_patches=16,    # více středů = lepší pokrytí
        overlap_ratio=0.5  # přísnější filtrace okrajů
    )
    for originals, padded, lengths in tqdm(test_loader, desc="Testing"):
        originals = originals.to(DEVICE, non_blocking=True)
        padded = padded.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE)

        sampled_defected, _ = sample_farthest_points(
            padded,
            K=200000,
            lengths=lengths,
        )

        print(f"Sampled Defected shape: {sampled_defected.shape}")
        pred2 = wrapper(sampled_defected, target_n=originals.shape[1])
        plot_pointcloud_to_image(pred2, "patches_merged.png")
        plot_pointcloud_to_image(sampled_defected[0], "patches_merged1.png")
        plot_pointcloud_to_image(originals[0], "patches_gt.png")

        visualize_patches(xyz=sampled_defected, wrapper=wrapper, output_path=f"patches.png")
        break


    mean_cd = total_cd / len(test_loader)
    return mean_cd


mean_cd = evaluate()

print(f"[RESULT] Mean Chamfer Distance (fine): {mean_cd:.6f}")
