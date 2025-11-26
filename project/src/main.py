from dataset.downloader import KaggleDownloader, HuggingFaceDownloader
from dataset import ShapeNetV2Dataset
import dotenv
import os
import json

dotenv.load_dotenv()

dataset = ShapeNetV2Dataset(
    root="data/ShapeNet", split="train", categories=["airplane"]
)

# downloader = HuggingFaceDownloader(
#     "data/ShapeNetCore/raw",
#     "ShapeNet/ShapeNetCore",
#     token=os.getenv("HUGGINGFACE_TOKEN"),
# )
# downloader.download()

# from dataset.shapenet import ShapeNetDataset
# from dataset.augmented import AugmentedDataset
# from dataset.defects import (
#     Noise,
#     Rotate,
#     Scale,
#     Translate,
#     LocalDropout,
#     FloatingCluster,
#     OutlierPoints,
#     LargeMissingRegion,
#     BridgingArtifact,
#     SurfaceBridgingArtifact,
#     HairLikeNoise,
#     SurfaceFlattening,
#     SurfaceBulging,
#     AnisotropicStretchNoise,
# )
# from visualize.viewers import SampleViewer
# import polyscope as ps

# dataset = AugmentedDataset(
#     dataset=ShapeNetDataset(
#         root="data/ShapeNet", split="train", categories=["Airplane"]
#     ),
#     defects=[
#         AnisotropicStretchNoise(
#             radius=0.12,
#             stretch_factor=0.20,
#             max_regions=1,
#             jitter=0.001,
#         )
#     ],
# )


# def render_callback(sample):
#     ps.register_point_cloud(
#         "original",
#         sample.original,
#         radius=0.0025,
#         color=(0.0, 1.0, 0.0),
#     )
#     ps.register_point_cloud(
#         "defected",
#         sample.defected,
#         radius=0.0025,
#         color=(1.0, 0.0, 0.0),
#     )


# def text_callback(sample):
#     for defect, params in sample.log.items():
#         ps.imgui.Separator()
#         ps.imgui.TextColored((1.0, 1.0, 1.0, 1.0), f"Applied {defect}:")
#         for key, value in params.items():
#             ps.imgui.BulletText(f"{key}: {value}")


# viewer = SampleViewer(
#     dataset=dataset,
#     render_callback=render_callback,
#     text_callback=text_callback,
# )
# viewer.show()  # blocking call
