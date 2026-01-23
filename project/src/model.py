import torch
import os
from dotenv import load_dotenv
from visualize.viewer import SampleViewer
from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import LargeMissingRegion, LocalDropout
from models.pcn import PCNRepairNet
import polyscope as ps

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
        LargeMissingRegion(removal_fraction=0.3),
    ],
)


# # Visualize some samples from the dataset
# viewer = SampleViewer(dataset)
# viewer.show()

# Load the pre-trained model
model = PCNRepairNet(
    feat_dim=1024,
    num_coarse=1024,
    grid_size=4,
)
checkpoint = torch.load(CHECKPOINT_DIR + "/pcn_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

original, defected = dataset[9201]

original = torch.from_numpy(original).float().unsqueeze(0).to(device)
defected = torch.from_numpy(defected).float().unsqueeze(0).to(device)

from pytorch3d.loss import chamfer_distance

with torch.no_grad():
    coarse, completed = model(defected)  # ⬅️ ROZBALIT

    cd, _ = chamfer_distance(completed, defected)
    print(f"Chamfer Distance: {cd.item():.6f}")

print(original.shape, defected.shape, completed.shape)

ps.init()

ps.register_point_cloud(
    "original",
    original.squeeze(0).detach().cpu().numpy(),
    radius=0.0025,
    color=(0.0, 1.0, 0.0),
)
ps.register_point_cloud(
    "defected",
    defected.squeeze(0).detach().cpu().numpy(),
    radius=0.0025,
    color=(1.0, 0.0, 0.0),
)

ps.register_point_cloud(
    "completed",
    completed.squeeze(0).detach().cpu().numpy(),
    radius=0.0025,
    color=(0, 0.0, 1.0),
)

ps.show()
