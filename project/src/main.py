from dataset.shapenet import ShapeNetDataset
from dataset.augmented import AugmentedDataset
from dataset.defects import Noise, Rotate, Scale, Translate
from visualize.viewers import SampleViewer
import polyscope as ps

dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root="data/ShapeNet", split="train", categories=["Pistol"]),
    defects=[
        Noise(sigma=0.01),
        Rotate(z=45),
        Scale(factor=1.2),
        Translate(x=0.1, y=0.0, z=0.0),
    ],
)


def render_callback(sample):
    ps.register_point_cloud(
        "original",
        sample.original,
        radius=0.003,
        color=(0.0, 1.0, 0.0),
    )
    ps.register_point_cloud(
        "defected",
        sample.defected,
        radius=0.003,
        color=(1.0, 0.0, 0.0),
    )


def text_callback(sample):
    for defect, params in sample.log.items():
        ps.imgui.Separator()
        ps.imgui.TextColored((1.0, 1.0, 1.0, 1.0), f"Applied {defect}:")
        for key, value in params.items():
            ps.imgui.BulletText(f"{key}: {value}")


viewer = SampleViewer(
    dataset=dataset,
    render_callback=render_callback,
    text_callback=text_callback,
)
viewer.show()  # blocking call
