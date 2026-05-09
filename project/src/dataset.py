import argparse
import json
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
from dataset.wrapper import NormalizeWrapperDataset
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


def _parse_xyz_degrees(value: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            "Expected three comma-separated values for XYZ rotation, e.g. 0,0,90"
        )
    return float(parts[0]), float(parts[1]), float(parts[2])


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


def _parse_output_formats(value: str) -> List[str]:
    supported = {"png", "svg", "pdf"}
    raw = [v.strip().lower().lstrip(".") for v in value.split(",") if v.strip()]
    if not raw:
        raise ValueError("--export-formats must contain at least one format")

    invalid = [fmt for fmt in raw if fmt not in supported]
    if invalid:
        raise ValueError(f"Unsupported format(s) {invalid}. Choose from: png, svg, pdf")

    unique = []
    seen = set()
    for fmt in raw:
        if fmt not in seen:
            seen.add(fmt)
            unique.append(fmt)
    return unique


def _build_base_dataset(args, dataset_name: str):
    categories = _parse_csv(args.categories) if args.categories else None
    if dataset_name == "shapenet":
        base_dataset = ShapeNetDataset(
            root=os.path.join(args.data_root, "ShapeNetV2"),
            categories=categories,
        )
    elif dataset_name == "modelnet":
        base_dataset = ModelNetDataset(
            root=os.path.join(args.data_root, "ModelNet40"),
            categories=categories,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return base_dataset


def _build_dataset(
    args,
    dataset_name: Optional[str] = None,
    mode: Optional[str] = None,
):
    dataset_name = dataset_name or args.dataset
    mode = mode or args.mode

    base_dataset = _build_base_dataset(args, dataset_name)

    if mode == "pure":
        dataset = base_dataset
        if args.normalize:
            dataset = NormalizeWrapperDataset(dataset)
        return dataset

    creator = (
        create_advanced_reconstruction_dataset
        if mode == "advanced"
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


def _extract_sample_category(sample: Any) -> str:
    if sample is None:
        return ""

    if hasattr(sample, "category"):
        value = getattr(sample, "category")
        if value is not None and str(value).strip():
            return str(value).strip()

    if isinstance(sample, dict):
        value = sample.get("category", sample.get("text", ""))
        if value is not None and str(value).strip():
            return str(value).strip()

    if hasattr(sample, "text"):
        value = getattr(sample, "text")
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


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


def _build_description(
    category: str,
    log: Optional[dict],
    *,
    include_log: bool,
) -> str:
    parts = []
    if category:
        parts.append(f"category: {category}")
    if include_log:
        summary = _summarize_log(log)
        if summary:
            parts.append(summary)
    return " | ".join(parts)


def _resolve_run_dir(args: argparse.Namespace) -> Tuple[str, str]:
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def _resolve_output_stem(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    mode: str,
    suite_mode: bool,
) -> str:
    default_stem = f"{dataset_name}_{mode}"
    if suite_mode or not args.output_name:
        return default_stem

    stem, _ = os.path.splitext(args.output_name)
    return stem or default_stem


def _save_gallery_in_formats(
    run_dir: str,
    output_stem: str,
    formats: Sequence[str],
    *,
    pointclouds: Sequence[Any],
    dataset_name: str,
    sample_indices: Sequence[int],
    descriptions: Sequence[str],
    badge_labels: Sequence[Sequence[str]],
    config: GalleryConfig,
    seed: int,
) -> List[str]:
    output_paths = []
    for fmt in formats:
        out_path = os.path.join(run_dir, f"{output_stem}.{fmt}")
        save_dataset_gallery(
            pointclouds,
            out_path,
            dataset_name=dataset_name,
            sample_indices=sample_indices,
            descriptions=descriptions,
            badge_labels=badge_labels,
            config=config,
            seed=seed,
        )
        output_paths.append(out_path)
    return output_paths


def _collect_gallery_rows(
    dataset,
    chosen_indices: Sequence[int],
    *,
    mode: str,
    custom_labels: Optional[List[str]],
    show_defect_log: bool,
):
    pointclouds = []
    descriptions = []
    kept_indices = []
    badge_labels = []
    skipped = []

    for idx in chosen_indices:
        try:
            sample = dataset[idx]
            original, defected, log = _extract_sample_clouds(sample)
            category = _extract_sample_category(sample)
            original = _prepare_cloud_for_gallery(original)

            if mode == "pure":
                pointclouds.append([original])
                if custom_labels:
                    badge_labels.append([custom_labels[0]])
                else:
                    badge_labels.append(["Sample"])
                descriptions.append(
                    _build_description(
                        category=category,
                        log=None,
                        include_log=False,
                    )
                )
            else:
                defected = _prepare_cloud_for_gallery(defected)
                pointclouds.append([original, defected])
                descriptions.append(
                    _build_description(
                        category=category,
                        log=log,
                        include_log=show_defect_log,
                    )
                )
                badge_labels.append(
                    custom_labels if custom_labels else ["Original", "Defected"]
                )

            kept_indices.append(idx)
        except Exception as exc:
            skipped.append({"index": int(idx), "error": str(exc)})
            print(f"[WARN] Skipping sample {idx}: {exc}")

    return pointclouds, descriptions, kept_indices, badge_labels, skipped


def _build_suite_specs(args: argparse.Namespace):
    if not args.generate_suite:
        return [
            {
                "dataset": args.dataset,
                "mode": args.mode,
            }
        ]

    specs = []
    for dataset_name in ("shapenet", "modelnet"):
        for mode in ("pure", "basic", "advanced"):
            specs.append({"dataset": dataset_name, "mode": mode})
    return specs


def _build_summary_payload(
    args: argparse.Namespace, run_name: str, formats: Sequence[str]
):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "output_root": args.output_dir,
        "requested": {
            "generate_suite": bool(args.generate_suite),
            "num_samples": int(args.num_samples),
            "sample_indices": args.sample_indices,
            "seed": int(args.seed),
            "categories": args.categories,
            "views": args.views,
            "point_rotation": args.point_rotation,
            "formats": list(formats),
        },
        "augmentation": {
            "defect_augmentation_count": int(args.defect_augmentation_count),
            "local_dropout_regions": int(args.local_dropout_regions),
            "dense": bool(args.dense),
            "dense_num_points": int(args.dense_num_points),
            "normalize": bool(args.normalize),
            "show_defect_log": bool(args.show_defect_log),
        },
        "galleries": [],
    }


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
        choices=["pure", "basic", "advanced"],
        help="Choose corruption pipeline used by core dataset builders.",
    )
    parser.add_argument(
        "--generate-suite",
        action="store_true",
        help="Generate full gallery suite: shapenet/modelnet x pure/basic/advanced.",
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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name under --output-dir (auto-generated when omitted).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="png",
        choices=["png", "svg", "pdf"],
        help="Backward-compatible single-format hint. By default, all formats are exported.",
    )
    parser.add_argument(
        "--export-formats",
        type=str,
        default="png,svg,pdf",
        help="Comma-separated output formats to export for each gallery (default: png,svg,pdf).",
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

    parser.add_argument(
        "--views",
        type=str,
        default="0,0;0,90",
        help="Semicolon-separated list of elev,azim pairs for views",
    )
    parser.add_argument("--point-size", type=float, default=4.5)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Camera zoom factor (>1 zooms in, <1 zooms out)",
    )
    parser.add_argument(
        "--point-rotation",
        type=str,
        default="90,0,0",
        help="Object-space XYZ rotation in degrees applied before rendering (e.g. 0,0,90).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--min-figure-width",
        type=float,
        default=0,
        help="Minimum gallery figure width in inches (important for SVG export size).",
    )
    parser.add_argument(
        "--min-figure-height",
        type=float,
        default=0,
        help="Minimum gallery figure height in inches (important for SVG export size).",
    )
    parser.add_argument(
        "--badge-fontsize",
        type=float,
        default=16.0,
        help="Badge label font size in points.",
    )
    parser.add_argument(
        "--caption-fontsize",
        type=float,
        default=14.0,
        help="Caption font size in points.",
    )
    parser.add_argument(
        "--side-note-fontsize",
        type=float,
        default=14.0,
        help="Side note font size in points.",
    )
    parser.add_argument(
        "--border-linewidth",
        type=float,
        default=0.2,
        help="Sample-card border width.",
    )
    parser.add_argument(
        "--block-view-width",
        type=float,
        default=3.0,
        help="Width contribution (inches) per rendered view in a sample block.",
    )
    parser.add_argument(
        "--block-row-height",
        type=float,
        default=3.0,
        help="Height contribution (inches) per row in a sample block.",
    )
    parser.add_argument(
        "--page-padding",
        type=float,
        default=0,
        help="Normalized outer page padding in [0, 0.25].",
    )

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
    parser.add_argument(
        "--show-defect-log",
        action="store_true",
        help="Include defect log details in caption text.",
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

    formats = _parse_output_formats(args.export_formats)
    run_dir, run_name = _resolve_run_dir(args)
    summary = _build_summary_payload(args, run_name=run_name, formats=formats)

    if not args.generate_images:
        print("[INFO] Image generation disabled (--no-generate-images).")
        return

    custom_labels = _parse_labels(args.cloud_labels) if args.cloud_labels else None
    suite_specs = _build_suite_specs(args)

    config_kwargs = {
        "max_sample_cols": args.max_sample_cols,
        "views": _parse_views(args.views),
        "point_size": args.point_size,
        "max_points": args.max_points,
        "zoom": args.zoom,
        "point_rotation_deg": _parse_xyz_degrees(args.point_rotation),
        "dpi": args.dpi,
        "badge_fontsize": args.badge_fontsize,
        "caption_fontsize": args.caption_fontsize,
        "side_note_fontsize": args.side_note_fontsize,
        "border_linewidth": args.border_linewidth,
        "block_view_width": args.block_view_width,
        "block_row_height": args.block_row_height,
        "min_figure_width": args.min_figure_width,
        "min_figure_height": args.min_figure_height,
        "outer_wspace": 0,
        "outer_hspace": 0,
        "page_padding": args.page_padding,
        "wrap_caption_chars": 80,
        "image_grid_cols": args.image_grid_cols,
        "image_grid_max_images": args.image_grid_max_images,
        "image_row_height_ratio": args.image_row_height,
    }

    config = GalleryConfig(**config_kwargs)

    generated_count = 0
    for spec in suite_specs:
        spec_dataset = spec["dataset"]
        spec_mode = spec["mode"]

        dataset = _build_dataset(
            args,
            dataset_name=spec_dataset,
            mode=spec_mode,
        )
        if len(dataset) == 0:
            print(f"[WARN] Skipping empty dataset for {spec_dataset}/{spec_mode}")
            continue

        if args.open_viewer:
            print(
                "[INFO] Closed Polyscope viewer, continuing with gallery generation..."
            )

        chosen_indices = _choose_indices(len(dataset), args)
        (
            pointclouds,
            descriptions,
            kept_indices,
            badge_labels,
            skipped,
        ) = _collect_gallery_rows(
            dataset,
            chosen_indices,
            mode=spec_mode,
            custom_labels=custom_labels,
            show_defect_log=args.show_defect_log,
        )

        if not pointclouds:
            print(f"[WARN] No valid samples for {spec_dataset}/{spec_mode}")
            continue

        stem = _resolve_output_stem(
            args,
            dataset_name=spec_dataset,
            mode=spec_mode,
            suite_mode=args.generate_suite,
        )
        output_paths = _save_gallery_in_formats(
            run_dir,
            output_stem=stem,
            formats=formats,
            pointclouds=pointclouds,
            dataset_name=f"{spec_dataset.capitalize()} ({spec_mode})",
            sample_indices=kept_indices,
            descriptions=descriptions,
            badge_labels=badge_labels,
            config=config,
            seed=args.seed,
        )

        summary["galleries"].append(
            {
                "dataset": spec_dataset,
                "mode": spec_mode,
                "total_samples_in_dataset": int(len(dataset)),
                "requested_sample_count": int(len(chosen_indices)),
                "rendered_sample_count": int(len(pointclouds)),
                "sample_indices": [int(i) for i in kept_indices],
                "skipped": skipped,
                "outputs": [os.path.relpath(p, run_dir) for p in output_paths],
            }
        )
        generated_count += 1
        print(
            f"[INFO] Generated {spec_dataset}/{spec_mode} -> {', '.join(output_paths)}"
        )

    if generated_count == 0:
        raise RuntimeError(
            "No gallery was generated. Check dataset and filtering options."
        )

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[INFO] Saved run outputs to: {run_dir}")
    print(f"[INFO] Saved run summary to: {summary_path}")


if __name__ == "__main__":
    main()
