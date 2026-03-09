import argparse
import os
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv

from dataset import CO3DDataset, ModelNetDataset, PhotogrammetricDataset, ShapeNetDataset
from visualize.dataset_gallery import (
    GalleryConfig,
    _to_numpy_points,
    save_dataset_gallery,
)
from dataset import AugmentedDataset
from dataset.defect import LargeMissingRegion, LocalDropout, Noise, Combined

def _parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_views(value: str) -> List[Tuple[float, float]]:
    views = []
    for pair in value.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        elev_s, azim_s = pair.split(",")
        views.append((float(elev_s), float(azim_s)))
    if not views:
        raise ValueError("At least one view must be provided")
    return views


def _parse_indices(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--sample-indices was provided but no indices were parsed")

    indices = []
    for part in parts:
        indices.append(int(part))
    return indices


def _parse_labels(value: str) -> List[str]:
    labels = [p.strip() for p in value.split(",") if p.strip()]
    if not labels:
        raise ValueError("--cloud-labels was provided but no labels were parsed")
    return labels


def _build_dataset(args):
    if args.dataset == "shapenet":
        categories = _parse_csv(args.categories) if args.categories else None
        base = ShapeNetDataset(
            root=os.path.join(args.data_root, "ShapeNetV2"),
            categories=categories,
        )

        return AugmentedDataset(
            dataset=base,
            defects=[
                Combined(
                    [
                        LargeMissingRegion(removal_fraction=0.4),
                        LocalDropout(radius=0.05, regions=5, dropout_rate=0.8),
                        Noise(0.01),
                    ]
                )
                for _ in range(5)
            ],
            detailed=True,
        )
    
    if args.dataset == "modelnet":
        categories = _parse_csv(args.categories) if args.categories else None
        base = ModelNetDataset(
            root=os.path.join(args.data_root, "ModelNet40"),
            categories=categories,
        )

        return AugmentedDataset(
            dataset=base,
            defects=[
                Combined(
                    [
                        LargeMissingRegion(removal_fraction=0.4),
                        LocalDropout(radius=0.05, regions=5, dropout_rate=0.8),
                        Noise(0.01),
                    ]
                )
                for _ in range(5)
            ],
            detailed=True,
        )

    if args.dataset == "photogrammetric":
        base = CO3DDataset(
            root=os.path.join(args.data_root, "CO3D"),
            categories=_parse_csv(args.co3d_categories),
            samples_per_category=args.samples_per_category,
        )
        return PhotogrammetricDataset(
            dataset=base,
            frames_per_sample=args.frames_per_sample,
            frames_strategy=args.frames_strategy,
            detailed=True,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create bit-thesis ready image gallery of dataset point-cloud samples.",
    )
    parser.add_argument("--dataset", required=True, choices=["shapenet", "modelnet", "photogrammetric"])
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated explicit sample indices (overrides --num-samples and --seed)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="dataset_gallery.png")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None, help="Comma separated categories for dataset")

    parser.add_argument("--views", type=str, default="30,45;30,135", help="Semicolon-separated list of elev,azim pairs for views")
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--zoom", type=float, default=1.0, help="Camera zoom factor (>1 zooms in, <1 zooms out)")
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--co3d-categories", type=str, default="cup,bench,car")
    parser.add_argument("--samples-per-category", type=int, default=10)
    parser.add_argument("--frames-per-sample", type=int, default=10)
    parser.add_argument("--frames-strategy", type=str, default="uniform", choices=["uniform", "keyframe", "random"])
    parser.add_argument(
        "--cloud-labels",
        type=str,
        default=None,
        help="Comma-separated row labels for clouds in one sample (e.g. Original,Defected or Original,Defected,PCN,PoinTr)",
    )

    parser.add_argument("--max-sample-cols", type=int, default=3, help="Max number of columns per sample (for multi-cloud samples like Original+Defected)")
    parser.add_argument("--image-grid-cols", type=int, default=0, help="Columns for image/mask rows; 0 = auto-wrap")
    parser.add_argument("--image-grid-max-images", type=int, default=0, help="Maximum images shown in image/mask rows; 0 = all")
    parser.add_argument("--image-row-height", type=float, default=2, help="Relative height of image/mask rows inside a sample card")

    args = parser.parse_args()

    data_folder_path = os.getenv("DATA_FOLDER_PATH", "")
    args.data_root = args.data_root or os.path.join(data_folder_path, "data")

    dataset = _build_dataset(args)
    dataset_type = args.dataset
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after loading")

    # Better defaults for very dense photogrammetric point clouds with RGB colors.
    if dataset_type == "photogrammetric":
        if args.max_points == 8192:
            args.max_points = 120000
        if args.point_size == 1.5:
            args.point_size = 0.8

    if args.sample_indices:
        raw_indices = _parse_indices(args.sample_indices)
        invalid = [i for i in raw_indices if i < 0 or i >= len(dataset)]
        if invalid:
            raise ValueError(
                f"Invalid sample indices {invalid}. Valid range is [0, {len(dataset) - 1}]"
            )
        # Keep input order, remove duplicates.
        chosen_indices = list(dict.fromkeys(raw_indices))
    else:
        rng = np.random.default_rng(args.seed)
        k = min(args.num_samples, len(dataset))
        chosen_indices = sorted(rng.choice(len(dataset), size=k, replace=False).tolist())

    pointclouds = []
    descriptions = []
    kept_indices = []
    badge_labels = []

    custom_labels = _parse_labels(args.cloud_labels) if args.cloud_labels else None

    if(dataset_type == "photogrammetric"):
        for idx in chosen_indices:
            try:
                sample = dataset[idx]
                if sample is None:
                    continue

                if len(sample) >= 4:
                    unmasked, masked, selected_frame_paths, selected_mask_paths = sample
                else:
                    unmasked, masked = sample
                    selected_frame_paths = []
                    selected_mask_paths = []

                # Keep raw cloud objects (trimesh.PointCloud) so renderer can use per-point RGB colors.
                rows = [unmasked, masked]
                if selected_frame_paths:
                    rows.append({"type": "images", "paths": selected_frame_paths})

                if selected_mask_paths:
                    rows.append({"type": "images", "paths": selected_mask_paths})

                pointclouds.append(rows)
                descriptions.append("")

                if custom_labels:
                    badge_labels.append(custom_labels)
                else:
                    labels = ["Unmasked", "Masked"]

                    if selected_frame_paths:
                        labels.append("Images")

                    if selected_mask_paths:
                        labels.append("Masks")

                    badge_labels.append(labels)

                kept_indices.append(idx)
            except Exception as exc:
                print(f"[WARN] Skipping sample {idx}: {exc}")
    elif(dataset_type in ["shapenet", "modelnet"]):
        for idx in chosen_indices:
            try:
                original, defected, log = dataset[idx]
                original = _to_numpy_points(original)
                defected = _to_numpy_points(defected)
                pointclouds.append([original, defected])
                descriptions.append("")
                badge_labels.append(custom_labels if custom_labels else ["Original", "Defected"])
                kept_indices.append(idx)
            except Exception as exc:
                print(f"[WARN] Skipping sample {idx}: {exc}")

    if not pointclouds:
        raise RuntimeError("No valid samples found for visualization")

    config_kwargs = {
        "max_sample_cols": args.max_sample_cols,
        "views": _parse_views(args.views),
        "point_size": args.point_size,
        "max_points": args.max_points,
        "zoom": args.zoom,
        "dpi": args.dpi,
        "image_grid_cols": args.image_grid_cols,
        "image_grid_max_images": args.image_grid_max_images,
        "image_row_height_ratio": args.image_row_height,
    }

    config = GalleryConfig(**config_kwargs)

    save_dataset_gallery(
        pointclouds,
        args.output,
        dataset_name=args.dataset.capitalize(),
        sample_indices=kept_indices,
        descriptions=descriptions,
        badge_labels=badge_labels,
        config=config,
        seed=args.seed,
    )

    print(f"[INFO] Saved gallery image to {args.output}")
    requested_info = (
        f"explicit indices ({len(chosen_indices)})"
        if args.sample_indices
        else f"random samples requested: {args.num_samples}"
    )
    print(f"[INFO] Dataset: {args.dataset} | {requested_info} | samples rendered: {len(pointclouds)}")
    print(f"[INFO] Indices: {kept_indices}")


if __name__ == "__main__":
    main()
