import torch
import os
from dotenv import load_dotenv
from visualize.viewer import SampleViewer
from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import LargeMissingRegion, LocalDropout
from models import PCN
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
        LargeMissingRegion(removal_fraction=0.1),
    ],
)


# # Visualize some samples from the dataset
# viewer = SampleViewer(dataset)
# viewer.show()

# Load the pre-trained model
model = PCN(num_dense=16384, latent_dim=1024, grid_size=4)
checkpoint = torch.load(CHECKPOINT_DIR + "/pcn_v2_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print(checkpoint["val_loss"])

original, defected = dataset[0]

original = original.unsqueeze(0).to(device)
defected = defected.float().unsqueeze(0).to(device)

from pytorch3d.loss import chamfer_distance

with torch.no_grad():
    coarse, completed = model(defected)  # ⬅️ ROZBALIT

    cd, _ = chamfer_distance(defected, original)
    print(f"Chamfer Distance: {cd.item():.6f}")

