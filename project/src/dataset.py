import argparse
import os
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv

from core.datasets import (
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
)
from dataset import ModelNetDataset, ShapeNetDataset
from visualize.dataset_gallery import (
    GalleryConfig,
    _to_numpy_points,
    save_dataset_gallery,
)


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
    categories = _parse_csv(args.categories) if args.categories else None
    if args.dataset == "shapenet":
        base_dataset = ShapeNetDataset(
            root=os.path.join(args.data_root, "ShapeNetV2"),
            categories=categories,
        )
    elif args.dataset == "modelnet":
        base_dataset = ModelNetDataset(
            root=os.path.join(args.data_root, "ModelNet40"),
            categories=categories,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    creator = (
        create_advanced_reconstruction_dataset
        if args.mode == "advanced"
        else create_basic_reconstruction_dataset
    )

    return creator(
        base_dataset=base_dataset,
        seed=args.seed,
        defect_augmentation_count=args.defect_augmentation_count,
        local_dropout_regions=args.local_dropout_regions,
        dense=args.dense,
        dense_root=args.dense_root,
        dense_num_points=args.dense_num_points,
        normalize=args.normalize,
        visualize=args.open_viewer,
        split_into_patches=args.split_into_patches,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        normalize_patches=args.normalize_patches,
        overlap_ratio=args.overlap_ratio,
        max_extra_patches=args.max_extra_patches,
    )


def _prepare_cloud_for_gallery(cloud: Any) -> Optional[np.ndarray]:
    arr = _to_numpy_points(cloud)
    if arr is None:
        return None

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.reshape(-1, 3)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected cloud shape (N, 3), got {arr.shape}")

    return arr


def _extract_sample_clouds(sample: Any) -> Tuple[Any, Any, Optional[dict]]:
    if sample is None:
        raise ValueError("Sample is None")

    if hasattr(sample, "original_pos") and hasattr(sample, "defected_pos"):
        log = getattr(sample, "log", None)
        return sample.original_pos, sample.defected_pos, log

    if isinstance(sample, dict):
        if "original_pos" in sample and "defected_pos" in sample:
            return sample["original_pos"], sample["defected_pos"], sample.get("log")
        if "pos" in sample:
            return sample["pos"], sample["pos"], sample.get("log")

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        maybe_log = (
            sample[2] if len(sample) >= 3 and isinstance(sample[2], dict) else None
        )
        return sample[0], sample[1], maybe_log

    if hasattr(sample, "pos"):
        return sample.pos, sample.pos, None

    raise TypeError(f"Unsupported sample type: {type(sample).__name__}")


def _summarize_log(log: Optional[dict]) -> str:
    if not log:
        return ""

    chunks = []
    for defect_name, params in log.items():
        if isinstance(params, dict) and params:
            kv = ", ".join(f"{k}={v}" for k, v in params.items())
            chunks.append(f"{defect_name}({kv})")
        else:
            chunks.append(str(defect_name))
    return " | ".join(chunks)


def _resolve_output_path(args: argparse.Namespace) -> str:
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name:
        filename = args.output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.dataset}_{args.mode}_{timestamp}.png"

    return os.path.join(args.output_dir, filename)


def _choose_indices(dataset_len: int, args: argparse.Namespace) -> List[int]:
    if args.sample_indices:
        raw_indices = _parse_indices(args.sample_indices)
        invalid = [i for i in raw_indices if i < 0 or i >= dataset_len]
        if invalid:
            raise ValueError(
                f"Invalid sample indices {invalid}. Valid range is [0, {dataset_len - 1}]"
            )
        return list(dict.fromkeys(raw_indices))

    rng = np.random.default_rng(args.seed)
    k = min(args.num_samples, dataset_len)
    return sorted(rng.choice(dataset_len, size=k, replace=False).tolist())


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Visualize reconstruction datasets with basic/advanced corruption modes.",
    )
    parser.add_argument("--dataset", required=True, choices=["shapenet", "modelnet"])
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "advanced"],
        help="Choose corruption pipeline used by core dataset builders.",
    )

    parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Open interactive Polyscope SampleViewer.",
    )
    parser.add_argument(
        "--generate-images",
        action="store_true",
        default=True,
        help="Generate gallery image output.",
    )
    parser.add_argument(
        "--no-generate-images",
        dest="generate_images",
        action="store_false",
        help="Disable image generation and only run viewer if enabled.",
    )

    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated explicit sample indices (overrides --num-samples and --seed)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dataset",
        help="Output directory for gallery image.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename inside --output-dir (auto-generated when omitted).",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma separated categories for dataset",
    )

    parser.add_argument("--defect-augmentation-count", type=int, default=5)
    parser.add_argument("--local-dropout-regions", type=int, default=5)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dense-root", type=str, default=None)
    parser.add_argument("--dense-num-points", type=int, default=100000)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--split-into-patches", action="store_true")
    parser.add_argument("--patch-size", type=int, default=8192)
    parser.add_argument("--num-patches", type=int, default=None)
    parser.add_argument("--normalize-patches", action="store_true")
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-extra-patches", type=int, default=None)

    parser.add_argument(
        "--views",
        type=str,
        default="30,45;30,135",
        help="Semicolon-separated list of elev,azim pairs for views",
    )
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Camera zoom factor (>1 zooms in, <1 zooms out)",
    )
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument(
        "--cloud-labels",
        type=str,
        default=None,
        help="Comma-separated row labels for clouds in one sample (e.g. Original,Defected or Original,Defected,PCN,PoinTr)",
    )

    parser.add_argument(
        "--max-sample-cols",
        type=int,
        default=3,
        help="Max number of columns per sample (for multi-cloud samples like Original+Defected)",
    )
    parser.add_argument(
        "--image-grid-cols",
        type=int,
        default=0,
        help="Columns for image/mask rows; 0 = auto-wrap",
    )
    parser.add_argument(
        "--image-grid-max-images",
        type=int,
        default=0,
        help="Maximum images shown in image/mask rows; 0 = all",
    )
    parser.add_argument(
        "--image-row-height",
        type=float,
        default=2,
        help="Relative height of image/mask rows inside a sample card",
    )

    args = parser.parse_args()

    ROOT_DIR = os.getenv("ROOT_DIR", "")
    args.data_root = args.data_root or os.path.join(ROOT_DIR, "data")
    args.dense_root = args.dense_root or os.path.join(
        args.data_root, "ShapeNetV2_dense"
    )

    if not args.open_viewer and not args.generate_images:
        raise ValueError(
            "Both viewer and image generation are disabled. Nothing to do."
        )

    dataset = _build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after loading")

    if args.open_viewer:
        print("[INFO] Closed Polyscope viewer, continuing with gallery generation...")

    if not args.generate_images:
        print("[INFO] Image generation disabled (--no-generate-images).")
        return

    chosen_indices = _choose_indices(len(dataset), args)

    pointclouds = []
    descriptions = []
    kept_indices = []
    badge_labels = []

    custom_labels = _parse_labels(args.cloud_labels) if args.cloud_labels else None

    for idx in chosen_indices:
        try:
            sample = dataset[idx]
            original, defected, log = _extract_sample_clouds(sample)
            original = _prepare_cloud_for_gallery(original)
            defected = _prepare_cloud_for_gallery(defected)

            pointclouds.append([original, defected])
            descriptions.append(_summarize_log(log))
            badge_labels.append(
                custom_labels if custom_labels else ["Original", "Defected"]
            )
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
    output_path = _resolve_output_path(args)

    save_dataset_gallery(
        pointclouds,
        output_path,
        dataset_name=f"{args.dataset.capitalize()} ({args.mode})",
        sample_indices=kept_indices,
        descriptions=descriptions,
        badge_labels=badge_labels,
        config=config,
        seed=args.seed,
    )

    print(f"[INFO] Saved gallery image to {output_path}")
    requested_info = (
        f"explicit indices ({len(chosen_indices)})"
        if args.sample_indices
        else f"random samples requested: {args.num_samples}"
    )
    print(
        f"[INFO] Dataset: {args.dataset} | Mode: {args.mode} | {requested_info} | samples rendered: {len(pointclouds)}"
    )
    print(f"[INFO] Indices: {kept_indices}")


if __name__ == "__main__":
    main()
