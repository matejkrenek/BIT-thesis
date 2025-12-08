from dataset import ShapeNetV2Dataset
from visualize.viewers import SampleViewer
import polyscope as ps
from dotenv import load_dotenv
import open3d as o3d

load_dotenv()

dataset = ShapeNetV2Dataset(root="data/ShapeNetV2", categories=["airplane"])


def render_callback(sample):
    ps.register_point_cloud(
        "pointcloud original",
        sample.pos,
        radius=0.0025,
        color=(0.0, 1.0, 0.0),
    )
    ps.register_surface_mesh(
        "mesh original",
        sample.mesh_pos.numpy(),
        sample.face.t().numpy(),
    )


viewer = SampleViewer(
    dataset=dataset,
    render_callback=render_callback,
)
viewer.show()  # blocking call
