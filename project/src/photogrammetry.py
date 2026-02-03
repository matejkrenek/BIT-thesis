import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import torch

from mini_dust3r.api import (
    OptimizedResult,
    inferece_dust3r,
    log_optimized_result,
)
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.utils.image import load_images
import polyscope as ps
import numpy as np


def extract_from_trimesh_pc(pc):
    points = np.asarray(pc.vertices, dtype=np.float32)

    colors = np.asarray(pc.colors)
    if colors.shape[1] == 4:  # RGBA → RGB
        colors = colors[:, :3]

    # normalize for Polyscope
    if colors.max() > 1.0:
        colors = colors.astype(np.float32) / 255.0

    return points, colors


def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=image_dir,
        model=model,
        device=device,
        batch_size=1,
    )

    optimized_results.point_cloud.export(Path("reconstruction.ply"))

    print(optimized_results)


if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.init("mini-dust3r", spawn=False)
    main(args.image_dir)
