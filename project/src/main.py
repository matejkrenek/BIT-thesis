from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import LocalDropout, FloatingCluster
from visualize.viewer import SampleViewer
from dotenv import load_dotenv
import open3d as o3d

# Suppress Open3D log messages and load environment variables
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
load_dotenv()

# Create an augmented dataset from shape net
dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root="data/ShapeNetV2"),
    defects=[
        LocalDropout(radius=0.05, regions=3, dropout_rate=0.75),
        FloatingCluster(
            cluster_radius=0.5, clusters=2, points_per_cluster=30, offset_factor=1.5
        ),
    ],
)

# Initialize and show the dataset sample viewer
viewer = SampleViewer(dataset=dataset)
viewer.show()
