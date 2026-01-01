from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defect import (
    LocalDropout,
    LargeMissingRegion,
    OutlierPoints,
    SurfaceFlattening,
)
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
        LargeMissingRegion(removal_fraction=0.1),
        OutlierPoints(num_points=50, scale_factor=2.0, mode="gaussian"),
        SurfaceFlattening(radius=0.1, plane_jitter=0.0002, max_regions=1),
    ],
)

# Initialize and show the dataset sample viewer
viewer = SampleViewer(dataset=dataset)
viewer.show()
