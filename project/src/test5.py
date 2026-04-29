# Photogrammetric pipeline for a folder of images
import sys
import os
from pathlib import Path
import trimesh
import polyscope as ps
import numpy as np
import torch
from mini_dust3r.api import inferece_dust3r
from mini_dust3r.model import AsymmetricCroCo3DStereo


def main():
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:
        image_folder = "./bunny_renders"  # Default folder
    image_folder = Path(image_folder)
    assert image_folder.exists(), f"Image folder {image_folder} does not exist!"

    # Collect image paths
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    images = sorted(
        [str(p) for p in image_folder.iterdir() if p.suffix.lower() in image_exts]
    )
    assert images, f"No images found in {image_folder}"

    # --- Load mini-dust3r model ---
    print("Loading mini-dust3r model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    # --- Run inference ---
    print("Running mini-dust3r inference...")
    optimized_results = inferece_dust3r(
        image_dir_or_list=images, model=model, device=device, batch_size=2, niter=200
    )
    pointcloud = optimized_results.point_cloud

    # --- Save to .ply file ---
    ply_path = image_folder / "duster_output.ply"
    print(f"Saving point cloud to {ply_path}")
    pointcloud.export(str(ply_path))

    # --- Visualize with polyscope ---
    print("Visualizing point cloud with polyscope...")
    ps.init()
    pts = np.asarray(pointcloud.vertices)
    if hasattr(pointcloud, "colors") and pointcloud.colors is not None:
        colors = np.asarray(pointcloud.colors)
        ps.register_point_cloud("DUSt3R Output", pts, enabled=True)
        ps.get_point_cloud("DUSt3R Output").add_color_quantity(
            "rgb", colors, defined_on="vertex", enabled=True
        )
    else:
        ps.register_point_cloud("DUSt3R Output", pts, enabled=True)
    ps.show()
    return


if __name__ == "__main__":
    main()
