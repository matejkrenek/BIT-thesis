from dataset import ShapeNetDataset, AugmentedDataset
from dataset.defects import (
    Noise,
    Rotate,
    Scale,
    Translate,
    LocalDropout,
    FloatingCluster,
    OutlierPoints,
    LargeMissingRegion,
    BridgingArtifact,
    SurfaceBridgingArtifact,
    HairLikeNoise,
    SurfaceFlattening,
    SurfaceBulging,
    AnisotropicStretchNoise,
)
from visualize.viewers import SampleViewer
import polyscope as ps

dataset = AugmentedDataset(
    dataset=ShapeNetDataset(
        root="data/ShapeNet", split="train", categories=["Airplane"]
    ),
    defects=[
        LargeMissingRegion(removal_fraction=0.2),
    ],
)


def render_callback(sample):
    ps.register_point_cloud(
        "original",
        sample.original,
        radius=0.0025,
        color=(0.0, 1.0, 0.0),
    )
    ps.register_point_cloud(
        "defected",
        sample.defected,
        radius=0.0025,
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
