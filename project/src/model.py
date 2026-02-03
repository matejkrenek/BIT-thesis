import torch
import os
from dotenv import load_dotenv
from visualize.viewer import SampleViewer
from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import (
    LargeMissingRegion,
    LocalDropout,
    Combined,
    Noise,
    Rotate,
    FloatingCluster,
)
from models import PCN
import polyscope as ps
import random as rnd

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"
CHECKPOINT_DIR = DATA_FOLDER_PATH + "/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create an augmented dataset from shape net
dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root=ROOT_DATA),
    defects=[
        Combined(
            [
                LargeMissingRegion(removal_fraction=rnd.uniform(0.1, 0.3)),
                LocalDropout(
                    radius=rnd.uniform(0.01, 0.1),
                    regions=5,
                    dropout_rate=rnd.uniform(0.5, 0.9),
                ),
            ]
        )
        for _ in range(10)
    ],
)


# # Visualize some samples from the dataset
# viewer = SampleViewer(dataset)
# viewer.show()
from pytorch3d.loss import chamfer_distance

# Load the pre-trained model
model = PCN(num_dense=16384, latent_dim=1024, grid_size=4)


checkpoint = torch.load(CHECKPOINT_DIR + "/exp1.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

original, defected = dataset[120123]

original = original.unsqueeze(0).to(device)
defected = defected.unsqueeze(0).to(device)

from pytorch3d.ops import sample_farthest_points

ps.init()

with torch.no_grad():
    coarse, completed = model(defected)  # ⬅️ ROZBALIT

    #     defected, _ = sample_farthest_points(
    #     padded,
    #     K=originals.shape[1],
    #     lengths=lengths,
    # )

    completed, _ = sample_farthest_points(
        completed,
        K=original.shape[1],
    )

    ps.register_point_cloud(
        "Defected Point Cloud",
        defected.squeeze(0).cpu().numpy(),
    )

    ps.register_point_cloud(
        "Completed Point Cloud",
        completed.squeeze(0).cpu().numpy(),
    )

    ps.register_point_cloud(
        "Original Point Cloud",
        original.squeeze(0).cpu().numpy(),
    )

    cd, _ = chamfer_distance(completed, original)
    print(f"Chamfer Distance: {cd.item():.6f}")

ps.show()
