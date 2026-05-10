from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from pytorch3d.ops import sample_farthest_points

from core import (
    ModelConfig,
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
    create_model,
    load_model_checkpoint,
)
from dataset import ModelNetDataset, ShapeNetDataset
from dataset.wrapper import NormalizeWrapperDataset, PatchWrapperDataset
from visualize.dataset_gallery import GalleryConfig, save_dataset_gallery


def _parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_views(value: str) -> List[Tuple[float, float]]:
    views: List[Tuple[float, float]] = []
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
        raise ValueError("Expected three comma-separated values for XYZ rotation")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_model_spec(value: str) -> Tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Invalid model spec format. Expected: name:model_type:/path/to/checkpoint"
        )
    name = parts[0].strip()
    model_type = parts[1].strip().lower()
    checkpoint = Path(parts[2].strip()).expanduser().resolve()
    if model_type not in {"pcn", "pointr", "adapointr"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return name, model_type, checkpoint


def _parse_output_formats(value: str) -> List[str]:
    supported = {"png", "svg", "pdf"}
    raw = [v.strip().lower().lstrip(".") for v in value.split(",") if v.strip()]
    if not raw:
        raise ValueError("--export-formats must contain at least one format")

    invalid = [fmt for fmt in raw if fmt not in supported]
    if invalid:
        raise ValueError(f"Unsupported format(s) {invalid}. Choose from: png, svg, pdf")

    unique: List[str] = []
    seen = set()
    for fmt in raw:
        if fmt not in seen:
            seen.add(fmt)
            unique.append(fmt)
    return unique


def _resolve_run_dir(args: argparse.Namespace) -> Tuple[Path, str]:
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def _build_summary_payload(
    args: argparse.Namespace,
    run_name: str,
    formats: Sequence[str],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "output_root": args.output_dir,
        "requested": {
            "action": args.action,
            "dataset": args.dataset,
            "mode": args.mode,
            "sample_index": int(args.sample_index),
            "seed": int(args.seed),
            "categories": args.categories,
            "views": args.views,
            "point_rotation": args.point_rotation,
            "formats": list(formats),
        },
        "patching": {
            "patch_size": int(args.patch_size),
            "num_patches": args.num_patches,
            "normalize_patches": bool(args.normalize_patches),
            "overlap_ratio": float(args.overlap_ratio),
            "max_extra_patches": args.max_extra_patches,
            "patching_method": args.patching_method,
            "patch_radius": float(args.patch_radius),
            "patch_center": args.patch_center,
            "patch_point_count_std": float(args.patch_point_count_std),
            "reassembled_points": int(args.reassembled_points),
        },
        "outputs": [],
    }


def _extract_sample_clouds(sample: Any) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(sample, "original_pos") and hasattr(sample, "defected_pos"):
        original = sample.original_pos
        defected = sample.defected_pos
    elif isinstance(sample, dict):
        original = sample.get("original_pos", sample.get("pos", None))
        defected = sample.get("defected_pos", sample.get("pos", None))
    else:
        raise TypeError(f"Unsupported sample type: {type(sample).__name__}")

    if original is None or defected is None:
        raise ValueError("Sample does not contain original/defected clouds")

    original_np = np.asarray(torch.as_tensor(original).float().cpu().numpy())
    defected_np = np.asarray(torch.as_tensor(defected).float().cpu().numpy())

    if original_np.ndim != 2 or original_np.shape[1] != 3:
        raise ValueError(f"original cloud must be (N,3), got {original_np.shape}")
    if defected_np.ndim != 2 or defected_np.shape[1] != 3:
        raise ValueError(f"defected cloud must be (N,3), got {defected_np.shape}")

    return original_np.astype(np.float32), defected_np.astype(np.float32)


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


def _build_category_caption(category: str) -> str:
    category = str(category).strip()
    return f"category: {category}" if category else ""


def _default_model_params(model_type: str) -> Dict[str, Any]:
    if model_type == "pcn":
        return {"num_dense": 16384, "latent_dim": 1024, "grid_size": 4}
    if model_type == "pointr":
        return {
            "trans_dim": 384,
            "knn_layer": 1,
            "num_pred": 16384,
            "num_query": 224,
        }
    if model_type == "adapointr":
        return {
            "num_query": 512,
            "num_points": 16384,
            "center_num": [512, 256],
            "global_feature_dim": 1024,
            "encoder_type": "graph",
            "decoder_type": "fc",
            "encoder_config": {
                "embed_dim": 384,
                "depth": 6,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "combine_style": "concat",
            },
            "decoder_config": {
                "embed_dim": 384,
                "depth": 8,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "self_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "cross_attn_combine_style": "concat",
            },
        }
    raise ValueError(f"Unsupported model type: {model_type}")


def _build_base_dataset(args: argparse.Namespace):
    categories = _parse_csv(args.categories) if args.categories else None
    if args.dataset == "shapenet":
        return ShapeNetDataset(
            root=os.path.join(args.data_root, "ShapeNetV2"), categories=categories
        )
    if args.dataset == "modelnet":
        return ModelNetDataset(
            root=os.path.join(args.data_root, "ModelNet40"), categories=categories
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _build_full_dataset(args: argparse.Namespace):
    base_dataset = _build_base_dataset(args)

    if args.mode == "pure":
        dataset = base_dataset
        if args.normalize:
            dataset = NormalizeWrapperDataset(dataset)
        return dataset

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
        visualize=False,
    )


def _build_patch_dataset(args: argparse.Namespace):
    full_dataset = _build_full_dataset(args)
    patch_dataset = PatchWrapperDataset(
        dataset=full_dataset,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        normalize_patches=args.normalize_patches,
        overlap_ratio=args.overlap_ratio,
        max_extra_patches=args.max_extra_patches,
        patching_method=args.patching_method,
        patch_radius=args.patch_radius,
        patch_center=args.patch_center,
        patch_point_count_std=args.patch_point_count_std,
        include_full_objects=True,
    )
    return full_dataset, patch_dataset


def _valid_patch_points(
    patches: np.ndarray,
    idx: int,
    valid_counts: Optional[np.ndarray],
) -> np.ndarray:
    patch = np.asarray(patches[idx], dtype=np.float32)
    if valid_counts is None:
        return patch
    count = int(valid_counts[idx])
    count = max(0, min(count, patch.shape[0]))
    return patch[:count]


def _restore_patch_coords(
    patch_points: np.ndarray,
    center: np.ndarray,
    *,
    method: str,
    patch_center: str,
    patch_radius: float,
) -> np.ndarray:
    return patch_points


def _reassemble_from_patches(
    patches: np.ndarray,
    centers: np.ndarray,
    *,
    valid_counts: Optional[np.ndarray],
    method: str,
    patch_center: str,
    patch_radius: float,
    target_points: int,
    seed: int,
) -> np.ndarray:
    pieces: List[np.ndarray] = []
    for i in range(int(patches.shape[0])):
        pts = _valid_patch_points(patches, i, valid_counts)
        if pts.size == 0:
            continue
        restored = _restore_patch_coords(
            pts,
            np.asarray(centers[i], dtype=np.float32),
            method=method,
            patch_center=patch_center,
            patch_radius=patch_radius,
        )
        pieces.append(restored.astype(np.float32, copy=False))

    if not pieces:
        return np.zeros((0, 3), dtype=np.float32)

    merged = np.concatenate(pieces, axis=0)

    if target_points > 0 and merged.shape[0] > target_points:
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(merged.shape[0], size=target_points, replace=False)
        merged = merged[sel]

    return merged.astype(np.float32, copy=False)


def _prepare_model_input(
    patch_points: np.ndarray,
    *,
    input_points: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(
        patch_points, dtype=torch.float32, device=device
    ).unsqueeze(0)
    target = max(1, int(input_points))

    if tensor.shape[1] > target:
        tensor, _ = sample_farthest_points(tensor, K=target)
    elif tensor.shape[1] < target and tensor.shape[1] > 0:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(
            int(tensor.shape[1]), size=target - int(tensor.shape[1]), replace=True
        )
        extra = tensor[:, torch.as_tensor(idx, device=device, dtype=torch.long), :]
        tensor = torch.cat([tensor, extra], dim=1)

    return tensor


def _predict_patch_completion(
    model: torch.nn.Module,
    model_type: str,
    patch_points: np.ndarray,
    *,
    input_points: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    in_t = _prepare_model_input(
        patch_points,
        input_points=input_points,
        seed=seed,
        device=device,
    )

    with torch.no_grad():
        if model_type in {"pcn", "pointr", "adapointr"}:
            _, pred = model(in_t)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    if isinstance(pred, (tuple, list)):
        pred = pred[-1]

    if not torch.is_tensor(pred) or pred.ndim != 3 or pred.shape[-1] != 3:
        raise RuntimeError("Unexpected completion output; expected tensor (B,N,3)")

    return pred[0].detach().cpu().numpy().astype(np.float32)


def _save_gallery_in_formats(
    run_dir: Path,
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
) -> List[Path]:
    output_paths: List[Path] = []
    for fmt in formats:
        out_path = run_dir / f"{output_stem}.{fmt}"
        save_dataset_gallery(
            pointclouds,
            str(out_path),
            dataset_name=dataset_name,
            sample_indices=sample_indices,
            descriptions=descriptions,
            badge_labels=badge_labels,
            config=config,
            seed=seed,
        )
        output_paths.append(out_path)
    return output_paths


def _open_viewer(clouds: Sequence[np.ndarray], labels: Sequence[str]) -> None:
    import polyscope as ps

    ps.init()
    for name, cloud in zip(labels, clouds):
        pts = np.asarray(cloud, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            continue
        ps.register_point_cloud(name, pts, point_render_mode="quad", radius=0.0022)
    ps.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Patch-based workflow: visualize extracted patches, reassemble patches into "
            "whole object, and optionally run completion on patches before reassembly."
        )
    )

    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["visualize_patches", "reassemble_patches", "complete_and_reassemble"],
    )

    parser.add_argument("--dataset", required=True, choices=["shapenet", "modelnet"])
    parser.add_argument(
        "--mode",
        type=str,
        default="advanced",
        choices=["pure", "basic", "advanced"],
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dense-root", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)

    parser.add_argument("--defect-augmentation-count", type=int, default=5)
    parser.add_argument("--local-dropout-regions", type=int, default=5)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dense-num-points", type=int, default=100000)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    parser.add_argument("--patch-size", type=int, default=8192)
    parser.add_argument("--num-patches", type=int, default=None)
    parser.add_argument("--normalize-patches", action="store_true")
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-extra-patches", type=int, default=None)
    parser.add_argument(
        "--patching-method",
        type=str,
        default="fps_knn",
        choices=["fps_knn"],
    )
    parser.add_argument("--patch-radius", type=float, default=0.05)
    parser.add_argument(
        "--patch-center",
        type=str,
        default="point",
        choices=["point", "mean", "none"],
    )
    parser.add_argument("--patch-point-count-std", type=float, default=0.0)

    parser.add_argument(
        "--max-visualized-patches",
        type=int,
        default=0,
        help="Maximum number of patches shown in visualize_patches action; 0 means all.",
    )
    parser.add_argument(
        "--reassembled-points",
        type=int,
        default=0,
        help="Optional downsample count after patch reassembly; 0 keeps all merged points.",
    )

    parser.add_argument(
        "--model-spec",
        type=str,
        default="",
        help=(
            "Required for complete_and_reassemble. Format: "
            "name:model_type:/path/to/checkpoint"
        ),
    )
    parser.add_argument(
        "--completion-input-points",
        type=int,
        default=2048,
        help="Number of input points fed to completion model per patch.",
    )

    parser.add_argument("--output-dir", type=str, default="outputs/patchbased")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name under --output-dir (auto-generated when omitted).",
    )
    parser.add_argument("--output-name", type=str, default="")
    parser.add_argument(
        "--output-format", type=str, default="png", choices=["png", "svg", "pdf"]
    )
    parser.add_argument(
        "--export-formats",
        type=str,
        default="png,svg,pdf",
        help="Comma-separated output formats to export for each gallery (default: png,svg,pdf).",
    )

    parser.add_argument("--open-viewer", action="store_true")
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
    parser.add_argument("--views", type=str, default="0,0;0,90")
    parser.add_argument("--point-size", type=float, default=4.5)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--point-rotation", type=str, default="90,0,0")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-figure-width", type=float, default=0)
    parser.add_argument("--min-figure-height", type=float, default=0)
    parser.add_argument("--badge-fontsize", type=float, default=16.0)
    parser.add_argument("--caption-fontsize", type=float, default=14.0)
    parser.add_argument("--side-note-fontsize", type=float, default=14.0)
    parser.add_argument("--border-linewidth", type=float, default=0.2)
    parser.add_argument("--block-view-width", type=float, default=3.0)
    parser.add_argument("--block-row-height", type=float, default=3.0)
    parser.add_argument("--page-padding", type=float, default=0)
    parser.add_argument("--max-sample-cols", type=int, default=4)
    parser.add_argument("--image-grid-cols", type=int, default=0)
    parser.add_argument("--image-grid-max-images", type=int, default=0)
    parser.add_argument("--image-row-height", type=float, default=2)

    return parser


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    root_dir = os.getenv("ROOT_DIR", "")
    args.data_root = args.data_root or os.path.join(root_dir, "data")
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

    full_dataset, patch_dataset = _build_patch_dataset(args)

    if args.sample_index < 0 or args.sample_index >= len(patch_dataset):
        raise IndexError(
            f"sample-index out of range: {args.sample_index} (dataset len={len(patch_dataset)})"
        )

    full_sample = full_dataset[args.sample_index]
    full_original, full_defected = _extract_sample_clouds(full_sample)
    sample_category = _extract_sample_category(full_sample)
    category_caption = _build_category_caption(sample_category)

    patch_sample = patch_dataset[args.sample_index]
    original_patches = np.asarray(
        torch.as_tensor(patch_sample.original_pos).float().cpu().numpy(),
        dtype=np.float32,
    )
    defected_patches = np.asarray(
        torch.as_tensor(patch_sample.defected_pos).float().cpu().numpy(),
        dtype=np.float32,
    )
    patch_centers = np.asarray(
        torch.as_tensor(patch_sample.patch_centers).float().cpu().numpy(),
        dtype=np.float32,
    )

    defected_valid_counts = None
    if hasattr(patch_sample, "defected_valid_counts"):
        defected_valid_counts = np.asarray(
            torch.as_tensor(patch_sample.defected_valid_counts).cpu().numpy()
        )

    model_name = ""
    model_type = ""
    completion_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action == "complete_and_reassemble":
        if not args.model_spec:
            raise ValueError("--model-spec is required for complete_and_reassemble")

        model_name, model_type, checkpoint = _parse_model_spec(args.model_spec)
        completion_model = create_model(
            ModelConfig(name=model_type, params=_default_model_params(model_type)),
            device=device,
        )
        load_model_checkpoint(
            checkpoint_path=checkpoint,
            model=completion_model,
            map_location=device,
            strict=False,
            weights_only=False,
        )
        completion_model.eval()

    clouds: List[np.ndarray] = []
    labels: List[str] = []
    original_patch_clouds: List[np.ndarray] = []
    defected_patch_clouds: List[np.ndarray] = []
    completed_patch_clouds: List[np.ndarray] = []

    if args.action == "visualize_patches":
        clouds.append(full_original)
        labels.append("Original full")
        clouds.append(full_defected)
        labels.append("Defected full")

        requested_patch_limit = int(args.max_visualized_patches)
        if requested_patch_limit <= 0:
            max_n = int(defected_patches.shape[0])
        else:
            max_n = max(1, requested_patch_limit)
        for patch_idx in range(min(max_n, int(defected_patches.shape[0]))):
            original_patch_pts = _valid_patch_points(
                original_patches, patch_idx, defected_valid_counts
            )
            original_patch_pts = _restore_patch_coords(
                original_patch_pts,
                patch_centers[patch_idx],
                method=args.patching_method,
                patch_center=args.patch_center,
                patch_radius=args.patch_radius,
            )
            patch_pts = _valid_patch_points(
                defected_patches, patch_idx, defected_valid_counts
            )
            patch_pts = _restore_patch_coords(
                patch_pts,
                patch_centers[patch_idx],
                method=args.patching_method,
                patch_center=args.patch_center,
                patch_radius=args.patch_radius,
            )
            clouds.append(original_patch_pts)
            labels.append(f"Original patch {patch_idx}")
            clouds.append(patch_pts)
            labels.append(f"Defected patch {patch_idx}")
            original_patch_clouds.append(original_patch_pts)
            defected_patch_clouds.append(patch_pts)

    elif args.action == "reassemble_patches":
        reassembled = _reassemble_from_patches(
            defected_patches,
            patch_centers,
            valid_counts=defected_valid_counts,
            method=args.patching_method,
            patch_center=args.patch_center,
            patch_radius=args.patch_radius,
            target_points=int(args.reassembled_points),
            seed=int(args.seed),
        )

        clouds.extend([full_original, full_defected, reassembled])
        labels.extend(["Original", "Defected", "Reassembled"])

    elif args.action == "complete_and_reassemble":
        completed_patches: List[np.ndarray] = []
        for patch_idx in range(int(defected_patches.shape[0])):
            patch_pts = _valid_patch_points(
                defected_patches, patch_idx, defected_valid_counts
            )
            if patch_pts.shape[0] == 0:
                continue
            completed_patch = _predict_patch_completion(
                completion_model,
                model_type,
                patch_pts,
                input_points=int(args.completion_input_points),
                seed=int(args.seed) + patch_idx,
                device=device,
            )
            completed_patches.append(completed_patch)

        requested_patch_limit = int(args.max_visualized_patches)
        if requested_patch_limit <= 0:
            max_n = len(completed_patches)
        else:
            max_n = max(1, requested_patch_limit)

        for patch_idx in range(min(max_n, int(defected_patches.shape[0]))):
            original_patch_pts = _valid_patch_points(
                original_patches, patch_idx, defected_valid_counts
            )
            patch_pts = _valid_patch_points(
                defected_patches, patch_idx, defected_valid_counts
            )
            completed_patch_pts = np.asarray(
                completed_patches[patch_idx], dtype=np.float32
            )

            original_patch_clouds.append(original_patch_pts)
            defected_patch_clouds.append(patch_pts)
            completed_patch_clouds.append(completed_patch_pts)

        if not completed_patches:
            raise RuntimeError("No completed patches were produced")

        max_len = max(p.shape[0] for p in completed_patches)
        completed_stack = np.zeros(
            (len(completed_patches), max_len, 3), dtype=np.float32
        )
        completed_counts = np.zeros((len(completed_patches),), dtype=np.int64)
        for i, patch in enumerate(completed_patches):
            completed_stack[i, : patch.shape[0]] = patch
            completed_counts[i] = patch.shape[0]

        completed_reassembled = _reassemble_from_patches(
            completed_stack,
            patch_centers[: len(completed_patches)],
            valid_counts=completed_counts,
            method=args.patching_method,
            patch_center=args.patch_center,
            patch_radius=args.patch_radius,
            target_points=int(args.reassembled_points),
            seed=int(args.seed),
        )

        defected_reassembled = _reassemble_from_patches(
            defected_patches,
            patch_centers,
            valid_counts=defected_valid_counts,
            method=args.patching_method,
            patch_center=args.patch_center,
            patch_radius=args.patch_radius,
            target_points=int(args.reassembled_points),
            seed=int(args.seed),
        )

        clouds.extend(
            [full_original, full_defected, defected_reassembled, completed_reassembled]
        )
        labels.extend(
            [
                "Original full",
                "Defected full",
                "Reassembled",
                f"Reassembled ({model_name})",
            ]
        )

    output_stem = (
        Path(args.output_name).stem
        if args.output_name
        else f"{args.dataset}_{args.mode}_{args.action}_sample_{args.sample_index:07d}"
    )
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

    output_paths: List[Path] = []
    if args.generate_images:
        gallery_pointclouds: List[List[np.ndarray]]
        gallery_badges: List[List[str]]
        gallery_descriptions: List[str]
        gallery_indices: List[int]

        if args.action == "visualize_patches":
            max_cols_for_patches = max(
                int(args.max_sample_cols), len(original_patch_clouds)
            )
            config.max_sample_cols = max_cols_for_patches

            gallery_pointclouds = [
                [full_original],
                [full_defected],
                original_patch_clouds,
                defected_patch_clouds,
            ]
            gallery_badges = [
                ["Original"],
                ["Defected"],
                [f"Original patch {i}" for i in range(len(original_patch_clouds))],
                [f"Defected patch {i}" for i in range(len(defected_patch_clouds))],
            ]
            gallery_descriptions = [
                category_caption,
                category_caption,
                category_caption,
                category_caption,
            ]
            gallery_indices = [
                int(args.sample_index),
                int(args.sample_index),
                int(args.sample_index),
                int(args.sample_index),
            ]
        elif args.action == "complete_and_reassemble":
            max_cols_for_patches = max(
                int(args.max_sample_cols), len(original_patch_clouds)
            )
            config.max_sample_cols = max_cols_for_patches

            gallery_pointclouds = [
                [full_original],
                [full_defected],
                original_patch_clouds,
                defected_patch_clouds,
                completed_patch_clouds,
                [defected_reassembled],
                [completed_reassembled],
            ]
            gallery_badges = [
                ["Original"],
                ["Defected"],
                [f"Original patch {i}" for i in range(len(original_patch_clouds))],
                [f"Defected patch {i}" for i in range(len(defected_patch_clouds))],
                [f"Completed patch {i}" for i in range(len(completed_patch_clouds))],
                ["Reassembled"],
                [f"Reassembled ({model_name})"],
            ]
            gallery_descriptions = [
                category_caption,
                category_caption,
                category_caption,
                category_caption,
                category_caption,
                category_caption,
                category_caption,
            ]
            gallery_indices = [
                int(args.sample_index),
                int(args.sample_index),
                int(args.sample_index),
                int(args.sample_index),
                int(args.sample_index),
                int(args.sample_index),
                int(args.sample_index),
            ]
        else:
            gallery_pointclouds = [list(clouds)]
            gallery_badges = [list(labels)]
            gallery_descriptions = [category_caption]
            gallery_indices = [int(args.sample_index)]

        output_paths = _save_gallery_in_formats(
            run_dir,
            output_stem=output_stem,
            formats=formats,
            pointclouds=gallery_pointclouds,
            dataset_name=f"{args.dataset.capitalize()} ({args.mode})",
            sample_indices=gallery_indices,
            descriptions=gallery_descriptions,
            badge_labels=gallery_badges,
            config=config,
            seed=args.seed,
        )

        summary["outputs"] = [
            os.path.relpath(str(p), str(run_dir)) for p in output_paths
        ]
        print(f"[INFO] Generated outputs: {', '.join(str(p) for p in output_paths)}")
    else:
        print("[INFO] Image generation disabled (--no-generate-images).")

    if args.open_viewer:
        _open_viewer(clouds, labels)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[INFO] Saved run outputs to: {run_dir}")
    print(f"[INFO] Saved run summary to: {summary_path}")


if __name__ == "__main__":
    main()
