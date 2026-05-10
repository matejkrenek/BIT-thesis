from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_farthest_points
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from core import (
    ModelConfig,
    bootstrap,
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
    create_model,
    create_train_val_test_dataloaders,
    get_default_model_params,
    load_model_checkpoint,
    logger,
)
from core.inference_pipeline import (
    MODEL_PRESETS,
    _apply_outlier_filter,
    _build_patch_dataset,
    _load_checkpoint_flexible,
    _predict_denoised_centers,
    _safe_first_patch_radius,
)
from core.cli_parsing import (
    parse_indices as _parse_indices,
    parse_optional_csv as _parse_csv,
    parse_views as _parse_views,
    parse_xyz_degrees as _parse_xyz_degrees,
)
from dataset import ModelNetDataset, ShapeNetDataset
from metrics import (
    chamfer_distance_metric,
    density_aware_chamfer_distance_metric,
    hausdorff_distance_metric,
)
from models import PCN, PoinTr
from visualize.dataset_gallery import GalleryConfig, save_dataset_gallery

SUPPORTED_METRICS = ("chamfer", "hausdorff", "dcd")


@dataclass(frozen=True)
class EvalModelSpec:
    name: str
    model_type: str
    checkpoint: Path


@dataclass
class PipelineRuntime:
    denoise_model: Optional[torch.nn.Module]
    outlier_model: Optional[torch.nn.Module]
    run_denoise: bool
    run_outlier_before: bool
    run_outlier_after: bool
    patch_radius: float
    points_per_patch: int
    batch_size: int
    workers: int
    cache_capacity: int
    n_neighbours: int
    outlier_threshold: float
    use_pca: bool
    patch_center: str
    point_tuple: int
    use_point_stn: bool
    fps_points: int


def _compute_metric_values_single(
    pred: torch.Tensor,
    gt: torch.Tensor,
    metrics: Sequence[str],
    density_alpha: float,
) -> Dict[str, float]:
    values = _compute_metric_values_batch(
        pred.unsqueeze(0),
        gt.unsqueeze(0),
        metrics=metrics,
        density_alpha=density_alpha,
    )
    return {k: float(v[0].item()) for k, v in values.items()}


def _parse_metrics(value: str) -> List[str]:
    metrics = [m.strip().lower() for m in value.split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics parsed from --metrics")

    invalid = [m for m in metrics if m not in SUPPORTED_METRICS]
    if invalid:
        raise ValueError(
            f"Unsupported metrics: {invalid}. Supported: {list(SUPPORTED_METRICS)}"
        )
    return metrics


def _validate_split_ratios(
    train_ratio: float, val_ratio: float
) -> Tuple[float, float, float]:
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be in range (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val-ratio must be in range [0, 1)")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0.0:
        raise ValueError(
            "Invalid split ratios: train_ratio + val_ratio must be < 1.0 so test split remains positive"
        )

    return train_ratio, val_ratio, test_ratio


def _parse_model_specs(values: Sequence[str]) -> List[EvalModelSpec]:
    if not values:
        raise ValueError("At least one --model-spec/--modelspec must be provided")

    specs: List[EvalModelSpec] = []
    for raw in values:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Invalid model spec format. Expected name:model_type:/path/to/checkpoint"
            )

        name = parts[0].strip()
        model_type = parts[1].strip().lower()
        checkpoint = parts[2].strip()

        if model_type not in {"pcn", "pointr", "adapointr"}:
            raise ValueError(f"Unsupported model_type '{model_type}' in '{raw}'")
        if not name:
            raise ValueError(f"Model name is empty in '{raw}'")
        if not checkpoint:
            raise ValueError(f"Checkpoint path is empty in '{raw}'")

        specs.append(
            EvalModelSpec(name=name, model_type=model_type, checkpoint=Path(checkpoint))
        )

    return specs


def _legacy_load_state_dict(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)

    model_state = model.state_dict()

    if any(k.startswith("module.") for k in state.keys()):
        stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
        if set(stripped.keys()) == set(model_state.keys()):
            state = stripped

    if not any(k.startswith("module.") for k in state.keys()) and any(
        k.startswith("module.") for k in model_state.keys()
    ):
        state = {f"module.{k}": v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(
            "Missing keys for model load ({count}): {keys}",
            count=len(missing),
            keys=missing[:5],
        )
    if unexpected:
        logger.warning(
            "Unexpected keys for model load ({count}): {keys}",
            count=len(unexpected),
            keys=unexpected[:5],
        )


def _build_model(spec: EvalModelSpec, device: torch.device) -> torch.nn.Module:
    model = create_model(
        ModelConfig(
            name=spec.model_type, params=get_default_model_params(spec.model_type)
        ),
        device=device,
    )

    try:
        load_model_checkpoint(
            checkpoint_path=spec.checkpoint,
            model=model,
            map_location=device,
            strict=True,
            weights_only=False,
        )
    except Exception as exc:
        logger.warning(
            "Core checkpoint loader failed for '{name}' ({err}); trying compatibility loader.",
            name=spec.name,
            err=str(exc),
        )
        _legacy_load_state_dict(model, spec.checkpoint, device)

    model.eval()
    return model


def _predict(
    model: torch.nn.Module,
    model_type: str,
    defected_batched: torch.Tensor,
    target_points: int,
) -> torch.Tensor:
    with torch.no_grad():
        if model_type == "pcn":
            _, pred = model(defected_batched)
        elif model_type in {"pointr", "adapointr"}:
            _, pred = model(defected_batched)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if pred.shape[1] != target_points:
            pred, _ = sample_farthest_points(pred, K=target_points)

    return pred


def _compute_metric_values_batch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    metrics: Sequence[str],
    density_alpha: float,
    pred_lengths: Optional[torch.Tensor] = None,
    gt_lengths: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    if "chamfer" in metrics:
        out["chamfer"] = chamfer_distance_metric(
            pred,
            gt,
            pred_lengths=pred_lengths,
            gt_lengths=gt_lengths,
            batch_reduction="none",
        )
    if "hausdorff" in metrics:
        out["hausdorff"] = hausdorff_distance_metric(
            pred,
            gt,
            pred_lengths=pred_lengths,
            gt_lengths=gt_lengths,
            reduction="none",
        )
    if "dcd" in metrics:
        out["dcd"] = density_aware_chamfer_distance_metric(
            pred,
            gt,
            pred_lengths=pred_lengths,
            gt_lengths=gt_lengths,
            alpha=density_alpha,
            reduction="none",
        )

    return out


def _compute_aggregate_table(
    records: List[Dict[str, object]], metrics: Sequence[str]
) -> List[Dict[str, object]]:
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for rec in records:
        model_name = str(rec["model"])
        metric_values = rec["metrics"]
        if model_name not in by_model:
            by_model[model_name] = {m: [] for m in metrics}
        for metric_name in metrics:
            if metric_name in metric_values:
                by_model[model_name][metric_name].append(
                    float(metric_values[metric_name])
                )

    rows: List[Dict[str, object]] = []
    for model_name, metric_map in by_model.items():
        for metric_name, values in metric_map.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "count": int(arr.shape[0]),
                }
            )

    rows.sort(key=lambda r: (str(r["model"]), str(r["metric"])))
    return rows


def _metric_columns(
    records: List[Dict[str, object]], base_metrics: Sequence[str]
) -> List[str]:
    extra: List[str] = []
    seen = set(base_metrics)
    for rec in records:
        metric_values = rec["metrics"]
        for key in metric_values.keys():
            if key in seen:
                continue
            seen.add(key)
            extra.append(str(key))

    return list(base_metrics) + sorted(extra)


def _build_segment_reference(
    original: torch.Tensor,
    defected: torch.Tensor,
    segment_threshold: float,
) -> Dict[str, torch.Tensor]:
    d2 = (
        knn_points(original.unsqueeze(0), defected.unsqueeze(0), K=1)
        .dists.squeeze(0)
        .squeeze(-1)
    )
    d = torch.sqrt(d2)

    repaired_target_mask = d > segment_threshold
    if repaired_target_mask.sum() == 0:
        repaired_target_mask[torch.argmax(d)] = True

    original_preserved_mask = ~repaired_target_mask
    if original_preserved_mask.sum() == 0:
        original_preserved_mask[torch.argmin(d)] = True
        repaired_target_mask = ~original_preserved_mask

    return {
        "repaired_target_mask": repaired_target_mask,
        "original_preserved_mask": original_preserved_mask,
        "repaired_target": original[repaired_target_mask],
        "original_preserved": original[original_preserved_mask],
    }


def _split_current_by_reference(
    current: torch.Tensor,
    original: torch.Tensor,
    repaired_target_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    nn_idx = (
        knn_points(current.unsqueeze(0), original.unsqueeze(0), K=1)
        .idx.squeeze(0)
        .squeeze(-1)
        .long()
    )
    hits_repaired = repaired_target_mask[nn_idx]

    return {
        "current_repaired": current[hits_repaired],
        "current_preserved": current[~hits_repaired],
    }


def _pair_metric(
    source: torch.Tensor,
    target: torch.Tensor,
    metric_name: str,
    density_alpha: float,
) -> float:
    if source.numel() == 0 or target.numel() == 0:
        return float("nan")

    if metric_name == "chamfer":
        return float(chamfer_distance_metric(source, target).item())
    if metric_name == "hausdorff":
        return float(hausdorff_distance_metric(source, target).item())
    if metric_name == "dcd":
        return float(
            density_aware_chamfer_distance_metric(
                source,
                target,
                alpha=density_alpha,
            ).item()
        )

    raise ValueError(f"Unsupported metric: {metric_name}")


def _compute_segmented_metrics(
    original: torch.Tensor,
    current: torch.Tensor,
    reference: Dict[str, torch.Tensor],
    current_split: Dict[str, torch.Tensor],
    metrics: Sequence[str],
    density_alpha: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for metric_name in metrics:
        out[f"repaired_target_vs_current_repaired_{metric_name}"] = _pair_metric(
            reference["repaired_target"],
            current_split["current_repaired"],
            metric_name=metric_name,
            density_alpha=density_alpha,
        )
        out[f"original_preserved_vs_current_preserved_{metric_name}"] = _pair_metric(
            reference["original_preserved"],
            current_split["current_preserved"],
            metric_name=metric_name,
            density_alpha=density_alpha,
        )

    return out


def _segmented_metric_lines(
    metric_values: Dict[str, float],
    prefix: str,
    metrics: Sequence[str],
) -> List[str]:
    lines: List[str] = []
    for metric_name in metrics:
        key = f"{prefix}_{metric_name}"
        if key in metric_values:
            lines.append(_format_metric_short(metric_name, float(metric_values[key])))
    return lines


def _save_per_sample_csv(
    path: Path, records: List[Dict[str, object]], metrics: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "model", *metrics])
        for rec in records:
            metric_values = rec["metrics"]
            writer.writerow(
                [
                    rec["sample_index"],
                    rec["model"],
                    *[metric_values.get(metric_name, "") for metric_name in metrics],
                ]
            )


def _save_aggregate_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "mean", "median", "std", "count"])
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["metric"],
                    row["mean"],
                    row["median"],
                    row["std"],
                    row["count"],
                ]
            )


def _format_metric_short(name: str, value: float) -> str:
    if name == "dcd":
        return f"DCD={value:.5f}"
    if name == "chamfer":
        return f"CD={value:.5f}"
    if name == "hausdorff":
        return f"HD={value:.5f}"
    return f"{name}={value:.5f}"


def _metrics_to_lines(
    metric_values: Dict[str, float], metrics: Sequence[str]
) -> List[str]:
    return [
        _format_metric_short(metric_name, metric_values[metric_name])
        for metric_name in metrics
        if metric_name in metric_values
    ]


def _build_dataset(args, data_root: Path):
    categories = _parse_csv(args.categories)

    if args.dataset == "shapenet":
        base_dataset = ShapeNetDataset(
            root=str(data_root / "ShapeNetV2"),
            categories=categories,
        )
    elif args.dataset == "modelnet":
        base_dataset = ModelNetDataset(
            root=str(data_root / "ModelNet40"),
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
        visualize=False,
    )


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


def _resolve_sample_category(
    *,
    full_dataset,
    test_dataset,
    sample_idx: int,
) -> str:
    try:
        sample = test_dataset[sample_idx]
        category = _extract_sample_category(sample)
        if category:
            return category
    except Exception:
        pass

    if isinstance(test_dataset, Subset) and hasattr(full_dataset, "dataset"):
        base_dataset = getattr(full_dataset, "dataset")
        global_idx = int(test_dataset.indices[sample_idx])
        try:
            base_item = base_dataset[global_idx]
            category = _extract_sample_category(base_item)
            if category:
                return category
        except Exception:
            pass

    return ""


def _build_category_caption(category: str) -> str:
    category = str(category).strip()
    return f"category: {category}" if category else ""


def _resolve_clean_gt_for_sample(
    *,
    full_dataset,
    test_dataset,
    sample_idx: int,
    fallback_gt: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(test_dataset, Subset):
        return fallback_gt

    if not hasattr(full_dataset, "dataset"):
        return fallback_gt

    base_dataset = getattr(full_dataset, "dataset")
    global_idx = int(test_dataset.indices[sample_idx])

    try:
        base_item = base_dataset[global_idx]
    except Exception:
        return fallback_gt

    if hasattr(base_item, "original_pos"):
        return torch.as_tensor(base_item.original_pos).float().to(fallback_gt.device)
    if isinstance(base_item, dict) and "original_pos" in base_item:
        return torch.as_tensor(base_item["original_pos"]).float().to(fallback_gt.device)

    return fallback_gt


def _build_pipeline_runtime(args, cfg) -> Optional[PipelineRuntime]:
    run_denoise = bool(args.run_denoise)
    run_outlier_before = bool(args.run_outlier_before)
    run_outlier_after = bool(args.run_outlier_after)

    if not (run_denoise or run_outlier_before or run_outlier_after):
        return None

    denoise_model = None
    outlier_model = None

    denoise_trainopt = None
    denoise_params_ckpt = (
        Path(args.denoise_params_checkpoint).expanduser().resolve()
        if args.denoise_params_checkpoint
        else None
    )
    if denoise_params_ckpt and denoise_params_ckpt.exists():
        denoise_trainopt = torch.load(
            denoise_params_ckpt,
            map_location="cpu",
            weights_only=False,
        )

    points_per_patch = int(args.pipeline_points_per_patch)
    denoise_default_params = get_default_model_params("pointcleannet")
    if points_per_patch <= 0:
        points_per_patch = int(
            getattr(
                denoise_trainopt,
                "points_per_patch",
                denoise_default_params["num_points"],
            )
        )

    patch_radius = float(args.pipeline_patch_radius)
    if patch_radius <= 0.0:
        patch_radius = _safe_first_patch_radius(
            getattr(denoise_trainopt, "patch_radius", [0.05]),
            0.05,
        )

    use_pca = bool(getattr(denoise_trainopt, "use_pca", False))
    patch_center = str(getattr(denoise_trainopt, "patch_center", "point"))
    point_tuple = int(getattr(denoise_trainopt, "point_tuple", 1))
    use_point_stn = bool(getattr(denoise_trainopt, "use_point_stn", True))
    use_feat_stn = bool(getattr(denoise_trainopt, "use_feat_stn", True))
    sym_op = str(getattr(denoise_trainopt, "sym_op", "max"))

    if run_denoise:
        if not args.denoise_model_checkpoint:
            raise ValueError("--run-denoise requires --denoise-model-checkpoint")

        denoise_ckpt = Path(args.denoise_model_checkpoint).expanduser().resolve()
        if not denoise_ckpt.exists():
            raise FileNotFoundError(f"Missing denoise checkpoint: {denoise_ckpt}")

        denoise_params = get_default_model_params("pointcleannet")
        denoise_params.update(
            {
                "num_points": int(points_per_patch),
                "use_point_stn": use_point_stn,
                "use_feat_stn": use_feat_stn,
                "sym_op": sym_op,
                "point_tuple": point_tuple,
            }
        )
        denoise_model = create_model(
            "pointcleannet",
            denoise_params,
            device=cfg.device,
        )
        _load_checkpoint_flexible(denoise_model, denoise_ckpt, cfg.device)
        denoise_model.eval()

    if run_outlier_before or run_outlier_after:
        if not args.outlier_model_checkpoint:
            raise ValueError("Outlier stages require --outlier-model-checkpoint")

        outlier_ckpt = Path(args.outlier_model_checkpoint).expanduser().resolve()
        if not outlier_ckpt.exists():
            raise FileNotFoundError(f"Missing outlier checkpoint: {outlier_ckpt}")

        outlier_params = get_default_model_params("pointcleannet_outliers")
        outlier_params.update(
            {
                "num_points": int(points_per_patch),
                "use_point_stn": use_point_stn,
                "use_feat_stn": use_feat_stn,
                "sym_op": sym_op,
                "point_tuple": point_tuple,
            }
        )
        outlier_model = create_model(
            "pointcleannet_outliers",
            outlier_params,
            device=cfg.device,
        )
        _load_checkpoint_flexible(outlier_model, outlier_ckpt, cfg.device)
        outlier_model.eval()

    return PipelineRuntime(
        denoise_model=denoise_model,
        outlier_model=outlier_model,
        run_denoise=run_denoise,
        run_outlier_before=run_outlier_before,
        run_outlier_after=run_outlier_after,
        patch_radius=float(patch_radius),
        points_per_patch=int(points_per_patch),
        batch_size=max(1, int(args.pipeline_batch_size)),
        workers=max(0, int(args.pipeline_workers)),
        cache_capacity=max(1, int(args.pipeline_cache_capacity)),
        n_neighbours=max(1, int(args.pipeline_n_neighbours)),
        outlier_threshold=float(args.pipeline_outlier_threshold),
        use_pca=use_pca,
        patch_center=patch_center,
        point_tuple=point_tuple,
        use_point_stn=use_point_stn,
        fps_points=max(1, int(args.pipeline_fps_points)),
    )


def _run_pipeline_preprocess(
    *,
    runtime: PipelineRuntime,
    defected_np: np.ndarray,
    clean_gt_np: np.ndarray,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    current = np.asarray(defected_np, dtype=np.float32)
    reference = np.asarray(clean_gt_np, dtype=np.float32)

    stages: Dict[str, np.ndarray] = {
        "input": current.copy(),
    }
    timings: Dict[str, float] = {}

    if runtime.run_outlier_before and runtime.outlier_model is not None:
        t0 = time.perf_counter()
        current, reference = _apply_outlier_filter(
            cloud_np=current,
            reference_np=reference,
            outlier_model=runtime.outlier_model,
            patch_radius=runtime.patch_radius,
            points_per_patch=runtime.points_per_patch,
            seed=int(seed),
            cache_capacity=runtime.cache_capacity,
            use_pca=runtime.use_pca,
            patch_center=runtime.patch_center,
            point_tuple=runtime.point_tuple,
            batch_size=runtime.batch_size,
            workers=runtime.workers,
            threshold=runtime.outlier_threshold,
            stage_label="eval_before",
        )
        timings["outlier_1"] = float(time.perf_counter() - t0)
        stages["outlier_1"] = current.copy()

    if runtime.run_denoise and runtime.denoise_model is not None:
        t0 = time.perf_counter()
        denoise_dataset = _build_patch_dataset(
            defected_np=current,
            original_np=reference,
            patch_radius=runtime.patch_radius,
            points_per_patch=runtime.points_per_patch,
            seed=int(seed),
            cache_capacity=runtime.cache_capacity,
            shape_name="eval",
            use_pca=runtime.use_pca,
            patch_center=runtime.patch_center,
            point_tuple=runtime.point_tuple,
        )
        denoise_loader = DataLoader(
            denoise_dataset,
            batch_size=runtime.batch_size,
            num_workers=runtime.workers,
        )
        shape_properties = _predict_denoised_centers(
            model=runtime.denoise_model,
            dataloader=denoise_loader,
            device=device,
            use_pca=runtime.use_pca,
            use_point_stn=runtime.use_point_stn,
            total_patches=int(denoise_dataset.shape_patch_count[0]),
            progress_desc="eval_denoise",
        )

        shp = denoise_dataset.shape_cache.get(0)
        pts = torch.tensor(shp.pts, dtype=torch.float32)
        n_nei = max(1, min(runtime.n_neighbours, int(pts.shape[0])))
        nearest_neighbours = torch.tensor(
            shp.kdtree.query(shp.pts, n_nei)[1], dtype=torch.long
        )
        displacement_vectors = shape_properties - pts
        mean_nei_disp = displacement_vectors[nearest_neighbours].mean(1)
        denoised_full = shape_properties - mean_nei_disp
        current = denoised_full.numpy().astype(np.float32)
        timings["denoise"] = float(time.perf_counter() - t0)
        stages["denoise"] = current.copy()

    if runtime.run_outlier_after and runtime.outlier_model is not None:
        t0 = time.perf_counter()
        current, reference = _apply_outlier_filter(
            cloud_np=current,
            reference_np=reference,
            outlier_model=runtime.outlier_model,
            patch_radius=runtime.patch_radius,
            points_per_patch=runtime.points_per_patch,
            seed=int(seed),
            cache_capacity=runtime.cache_capacity,
            use_pca=runtime.use_pca,
            patch_center=runtime.patch_center,
            point_tuple=runtime.point_tuple,
            batch_size=runtime.batch_size,
            workers=runtime.workers,
            threshold=runtime.outlier_threshold,
            stage_label="eval_after",
        )
        timings["outlier_2"] = float(time.perf_counter() - t0)
        stages["outlier_2"] = current.copy()

    t0 = time.perf_counter()
    current_t = torch.from_numpy(current).unsqueeze(0).to(device=device)
    fps_k = min(runtime.fps_points, int(current_t.shape[1]))
    fps_t, _ = sample_farthest_points(current_t, K=fps_k)
    stages["fps"] = fps_t[0].detach().cpu().numpy().astype(np.float32)
    timings["fps"] = float(time.perf_counter() - t0)

    return stages, timings


def _resolve_run_dir(args, default_output_root: Path) -> Path:
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = default_output_root / output_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_eval_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for model evaluation and gallery export."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models, save sample gallery, and export aggregate stats."
    )

    parser.add_argument("--dataset", required=True, choices=["shapenet", "modelnet"])
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "advanced"],
        help="Choose corruption pipeline used by core dataset builders.",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--categories", type=str, default=None, help="Comma-separated categories"
    )
    parser.add_argument(
        "--model-spec",
        "--modelspec",
        action="append",
        dest="model_specs",
        default=[],
        help="Format: name:model_type:/abs/or/rel/checkpoint.pt (repeatable)",
    )
    parser.add_argument("--metrics", type=str, default="chamfer,hausdorff,dcd")
    parser.add_argument(
        "--scenario",
        type=str,
        default="a",
        choices=["a", "b", "c", "all"],
        help="Evaluation scenario: a=completion on dataset input, b=pipeline stages, c=completion on pipeline output, all=run all scenarios.",
    )
    parser.add_argument("--density-alpha", type=float, default=1000.0)
    parser.add_argument("--segment-threshold", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio. Test ratio is computed implicitly as 1 - train_ratio - val_ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio. Test ratio is computed implicitly as 1 - train_ratio - val_ratio.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Optional number of samples to process from test split. Omit to process all test samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval",
        help="Output directory for evaluation runs. Relative paths are under OUTPUT_DIR.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run subdirectory name inside output-dir (auto-generated when omitted).",
    )
    parser.add_argument("--gallery-name", type=str, default="evaluation_gallery.png")
    parser.add_argument(
        "--metrics-csv-name", type=str, default="evaluation_per_sample.csv"
    )
    parser.add_argument(
        "--summary-csv-name", type=str, default="evaluation_summary.csv"
    )
    parser.add_argument(
        "--segmented-gallery-name", type=str, default="evaluation_segmented_gallery.png"
    )
    parser.add_argument(
        "--segmented-metrics-csv-name",
        type=str,
        default="evaluation_segmented_per_sample.csv",
    )
    parser.add_argument(
        "--segmented-summary-csv-name",
        type=str,
        default="evaluation_segmented_summary.csv",
    )
    parser.add_argument("--views", type=str, default="0,0;0,90")
    parser.add_argument("--point-size", type=float, default=4.5)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument(
        "--point-rotation",
        type=str,
        default="90,0,0",
        help="Object-space XYZ rotation in degrees applied before rendering (e.g. 0,0,90).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-figure-width", type=float, default=0)
    parser.add_argument("--min-figure-height", type=float, default=0)
    parser.add_argument("--badge-fontsize", type=float, default=16.0)
    parser.add_argument(
        "--metrics-fontsize",
        type=float,
        default=8.0,
        help="Font size for per-cloud metric detail text shown under badge labels.",
    )
    parser.add_argument("--caption-fontsize", type=float, default=14.0)
    parser.add_argument("--side-note-fontsize", type=float, default=14.0)
    parser.add_argument("--border-linewidth", type=float, default=0.2)
    parser.add_argument("--block-view-width", type=float, default=3.0)
    parser.add_argument("--block-row-height", type=float, default=3.0)
    parser.add_argument("--page-padding", type=float, default=0)
    parser.add_argument("--max-sample-cols", type=int, default=3)
    parser.add_argument("--defect-augmentation-count", type=int, default=5)
    parser.add_argument("--local-dropout-regions", type=int, default=5)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dense-root", type=str, default=None)
    parser.add_argument("--dense-num-points", type=int, default=100000)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--run-denoise", action="store_true", default=False)
    parser.add_argument("--run-outlier-before", action="store_true", default=False)
    parser.add_argument("--run-outlier-after", action="store_true", default=False)
    parser.add_argument(
        "--denoise-model-checkpoint",
        type=str,
        default="",
        help="Checkpoint path for pointcleannet denoising model.",
    )
    parser.add_argument(
        "--denoise-params-checkpoint",
        type=str,
        default=str(MODEL_PRESETS["denoise"]["params_checkpoint"]),
        help="Optional params checkpoint for denoise defaults.",
    )
    parser.add_argument(
        "--outlier-model-checkpoint",
        type=str,
        default="",
        help="Checkpoint path for pointcleannet_outliers model.",
    )
    parser.add_argument("--pipeline-points-per-patch", type=int, default=0)
    parser.add_argument("--pipeline-patch-radius", type=float, default=0.0)
    parser.add_argument("--pipeline-batch-size", type=int, default=128)
    parser.add_argument("--pipeline-workers", type=int, default=1)
    parser.add_argument("--pipeline-cache-capacity", type=int, default=100)
    parser.add_argument("--pipeline-n-neighbours", type=int, default=100)
    parser.add_argument("--pipeline-outlier-threshold", type=float, default=0.6)
    parser.add_argument("--pipeline-fps-points", type=int, default=10000)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = bootstrap(seed=int(args.seed), data_subdir="")

    metrics = _parse_metrics(args.metrics)
    scenario = str(args.scenario).lower()
    run_scenario_a = scenario in {"a", "all"}
    run_scenario_b = scenario in {"b", "all"}
    run_scenario_c = scenario in {"c", "all"}

    if run_scenario_a or run_scenario_c:
        model_specs = _parse_model_specs(args.model_specs)
    else:
        model_specs = _parse_model_specs(args.model_specs) if args.model_specs else []

    if (run_scenario_b or run_scenario_c) and args.mode != "advanced":
        logger.warning(
            "Scenarios B/C are designed for advanced dataset mode (current mode={mode}).",
            mode=args.mode,
        )

    data_root = (
        Path(args.data_root).expanduser().resolve() if args.data_root else cfg.data_dir
    )
    if args.dense_root:
        args.dense_root = str(Path(args.dense_root).expanduser().resolve())
    else:
        args.dense_root = str(data_root / "ShapeNetV2_dense")

    run_dir = _resolve_run_dir(args, cfg.output_dir)
    gallery_output = run_dir / args.gallery_name
    metrics_csv = run_dir / args.metrics_csv_name
    summary_csv = run_dir / args.summary_csv_name
    segmented_gallery_output = run_dir / args.segmented_gallery_name
    segmented_metrics_csv = run_dir / args.segmented_metrics_csv_name
    segmented_summary_csv = run_dir / args.segmented_summary_csv_name

    logger.info("Device: {device}", device=cfg.device)
    logger.info("Data root: {root}", root=data_root)
    logger.info("Run output dir: {run_dir}", run_dir=run_dir)

    train_ratio, val_ratio, test_ratio = _validate_split_ratios(
        args.train_ratio,
        args.val_ratio,
    )

    dataset = _build_dataset(args, data_root)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after loading")

    eval_batch_size = int(args.batch_size)
    if args.mode == "advanced" and eval_batch_size > 1:
        logger.warning(
            "Advanced dataset can contain variable-size original clouds; forcing eval batch size from {src} to 1.",
            src=eval_batch_size,
        )
        eval_batch_size = 1

    _, _, test_loader = create_train_val_test_dataloaders(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=eval_batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataset = test_loader.dataset
    test_split_size = len(test_dataset)
    logger.info(
        "Dataset split ratios train/val/test = {train:.4f}/{val:.4f}/{test:.4f}",
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
    )
    logger.info("Test split size: {size}", size=test_split_size)

    if args.test_samples is None:
        processed_test_size = test_split_size
    else:
        requested_test_samples = int(args.test_samples)
        if requested_test_samples < 1:
            raise ValueError("--test-samples must be at least 1 when provided")
        if requested_test_samples > test_split_size:
            raise ValueError(
                f"--test-samples={requested_test_samples} exceeds test split size ({test_split_size})"
            )
        processed_test_size = requested_test_samples

    logger.info(
        "Processing {count} test sample(s)",
        count=processed_test_size,
    )

    pipeline_runtime = _build_pipeline_runtime(args, cfg)
    if pipeline_runtime is None and (run_scenario_b or run_scenario_c):
        pipeline_runtime = PipelineRuntime(
            denoise_model=None,
            outlier_model=None,
            run_denoise=False,
            run_outlier_before=False,
            run_outlier_after=False,
            patch_radius=(
                float(args.pipeline_patch_radius)
                if float(args.pipeline_patch_radius) > 0
                else 0.05
            ),
            points_per_patch=max(
                1,
                (
                    int(args.pipeline_points_per_patch)
                    if int(args.pipeline_points_per_patch) > 0
                    else 500
                ),
            ),
            batch_size=max(1, int(args.pipeline_batch_size)),
            workers=max(0, int(args.pipeline_workers)),
            cache_capacity=max(1, int(args.pipeline_cache_capacity)),
            n_neighbours=max(1, int(args.pipeline_n_neighbours)),
            outlier_threshold=float(args.pipeline_outlier_threshold),
            use_pca=False,
            patch_center="point",
            point_tuple=1,
            use_point_stn=True,
            fps_points=max(1, int(args.pipeline_fps_points)),
        )

    if args.sample_indices:
        chosen_indices = _parse_indices(args.sample_indices)
        invalid = [i for i in chosen_indices if i < 0 or i >= processed_test_size]
        if invalid:
            raise ValueError(
                f"Invalid sample indices {invalid}. Valid range is [0, {processed_test_size - 1}]"
            )
        chosen_indices = list(dict.fromkeys(chosen_indices))
    else:
        rng = np.random.default_rng(args.seed)
        k = min(args.num_samples, processed_test_size)
        chosen_indices = sorted(
            rng.choice(processed_test_size, size=k, replace=False).tolist()
        )

    selected_set = set(chosen_indices)

    model_entries: List[Tuple[EvalModelSpec, torch.nn.Module]] = []
    for spec in model_specs:
        checkpoint_path = spec.checkpoint.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(
            "Loading model '{name}' ({mtype}) from {path}",
            name=spec.name,
            mtype=spec.model_type,
            path=checkpoint_path,
        )

        model_entries.append(
            (
                EvalModelSpec(
                    name=spec.name,
                    model_type=spec.model_type,
                    checkpoint=checkpoint_path,
                ),
                _build_model(
                    EvalModelSpec(
                        name=spec.name,
                        model_type=spec.model_type,
                        checkpoint=checkpoint_path,
                    ),
                    cfg.device,
                ),
            )
        )

    per_sample_records: List[Dict[str, object]] = []
    selected_payload: Dict[int, Dict[str, object]] = {}
    selected_pipeline_payload: Dict[int, Dict[str, object]] = {}
    segmented_records: List[Dict[str, object]] = []
    segmented_selected_payload: Dict[int, Dict[str, object]] = {}

    running_sample_idx = 0
    total_eval_batches = (
        (processed_test_size + eval_batch_size - 1) // eval_batch_size
        if processed_test_size > 0
        else 0
    )
    for batch in tqdm(
        test_loader,
        total=total_eval_batches,
        desc="Evaluating",
        unit="batch",
    ):
        if running_sample_idx >= processed_test_size:
            break

        originals, padded_defected, defected_lengths = batch
        if originals is None:
            continue

        originals = originals.to(cfg.device, non_blocking=True)
        padded_defected = padded_defected.to(cfg.device, non_blocking=True)
        defected_lengths = defected_lengths.to(cfg.device)

        batch_size_actual = originals.shape[0]
        remaining = processed_test_size - running_sample_idx
        if batch_size_actual > remaining:
            originals = originals[:remaining]
            padded_defected = padded_defected[:remaining]
            defected_lengths = defected_lengths[:remaining]
            batch_size_actual = int(remaining)

        batch_indices_cpu = list(
            range(running_sample_idx, running_sample_idx + batch_size_actual)
        )
        running_sample_idx += batch_size_actual

        target_points = originals.shape[1]
        defected_for_model, _ = sample_farthest_points(
            padded_defected,
            K=target_points,
            lengths=defected_lengths,
        )

        model_pred_batches: Dict[str, torch.Tensor] = {}
        model_pred_time_per_sample: Dict[str, float] = {}
        if run_scenario_a:
            for spec, model in model_entries:
                t_pred0 = time.perf_counter()
                pred = _predict(
                    model, spec.model_type, defected_for_model, target_points
                )
                pred_elapsed = float(time.perf_counter() - t_pred0)
                per_sample_elapsed = pred_elapsed / max(1, int(batch_size_actual))
                model_pred_time_per_sample[spec.name] = per_sample_elapsed
                model_pred_batches[spec.name] = pred.detach().cpu()

        for i, sample_idx in enumerate(batch_indices_cpu):
            original_i = originals[i]
            sample_category = _resolve_sample_category(
                full_dataset=dataset,
                test_dataset=test_dataset,
                sample_idx=sample_idx,
            )
            sample_category_caption = _build_category_caption(sample_category)
            if args.mode == "advanced":
                original_i = _resolve_clean_gt_for_sample(
                    full_dataset=dataset,
                    test_dataset=test_dataset,
                    sample_idx=sample_idx,
                    fallback_gt=original_i,
                )
            defected_i = defected_for_model[i]
            scenario_b_stage_metrics: Dict[str, Dict[str, float]] = {}

            if run_scenario_b or run_scenario_c:
                if pipeline_runtime is None:
                    raise RuntimeError(
                        "Internal error: pipeline_runtime is not initialized"
                    )

                defected_full_i = padded_defected[i, : int(defected_lengths[i].item())]
                pipeline_stages, pipeline_stage_timings = _run_pipeline_preprocess(
                    runtime=pipeline_runtime,
                    defected_np=defected_full_i.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    clean_gt_np=original_i.detach().cpu().numpy().astype(np.float32),
                    seed=int(args.seed) + int(sample_idx),
                    device=cfg.device,
                )

                for stage_name, seconds in pipeline_stage_timings.items():
                    per_sample_records.append(
                        {
                            "sample_index": sample_idx,
                            "model": f"T::{stage_name}",
                            "metrics": {"seconds": float(seconds)},
                        }
                    )

                gt_fps_points = min(
                    pipeline_runtime.fps_points, int(original_i.shape[0])
                )
                gt_fps_i, _ = sample_farthest_points(
                    original_i.unsqueeze(0),
                    K=gt_fps_points,
                )
                gt_fps_i = gt_fps_i[0]

                if run_scenario_b:
                    scenario_b_rows: List[Tuple[str, str]] = [("B::Input", "input")]
                    if "outlier_1" in pipeline_stages:
                        scenario_b_rows.append(("B::OutlierRemoval#1", "outlier_1"))
                    if "denoise" in pipeline_stages:
                        scenario_b_rows.append(("B::Denoising", "denoise"))
                    if "outlier_2" in pipeline_stages:
                        scenario_b_rows.append(("B::OutlierRemoval#2", "outlier_2"))

                    for row_name, stage_key in scenario_b_rows:
                        stage_t = (
                            torch.from_numpy(pipeline_stages[stage_key])
                            .float()
                            .to(cfg.device)
                        )
                        stage_metrics = _compute_metric_values_single(
                            pred=stage_t,
                            gt=original_i,
                            metrics=metrics,
                            density_alpha=args.density_alpha,
                        )
                        per_sample_records.append(
                            {
                                "sample_index": sample_idx,
                                "model": row_name,
                                "metrics": stage_metrics,
                            }
                        )
                        scenario_b_stage_metrics[stage_key] = stage_metrics

                    fps_t = (
                        torch.from_numpy(pipeline_stages["fps"]).float().to(cfg.device)
                    )
                    fps_metrics = _compute_metric_values_single(
                        pred=fps_t,
                        gt=gt_fps_i,
                        metrics=metrics,
                        density_alpha=args.density_alpha,
                    )
                    per_sample_records.append(
                        {
                            "sample_index": sample_idx,
                            "model": "B::FPS10k",
                            "metrics": fps_metrics,
                        }
                    )
                    scenario_b_stage_metrics["fps"] = fps_metrics
            else:
                pipeline_stages = {}
                pipeline_stage_timings = {}
                gt_fps_i = None

            sample_predictions: Dict[str, torch.Tensor] = {}
            sample_metrics: Dict[str, Dict[str, float]] = {}
            sample_segmented: Dict[str, Dict[str, object]] = {}
            sample_c_predictions: Dict[str, torch.Tensor] = {}
            sample_c_metrics: Dict[str, Dict[str, float]] = {}
            defected_metric_values: Dict[str, float] = {}

            if run_scenario_a:
                reference = _build_segment_reference(
                    original=original_i,
                    defected=defected_i,
                    segment_threshold=args.segment_threshold,
                )
                defected_metric_values = _compute_metric_values_single(
                    pred=defected_i,
                    gt=original_i,
                    metrics=metrics,
                    density_alpha=args.density_alpha,
                )
                per_sample_records.append(
                    {
                        "sample_index": sample_idx,
                        "model": "A::Defected",
                        "metrics": defected_metric_values,
                    }
                )

                for spec in model_specs:
                    model_metric_values = _compute_metric_values_single(
                        pred=model_pred_batches[spec.name][i].to(cfg.device),
                        gt=original_i,
                        metrics=metrics,
                        density_alpha=args.density_alpha,
                    )

                    per_sample_records.append(
                        {
                            "sample_index": sample_idx,
                            "model": f"T::A::{spec.name}",
                            "metrics": {
                                "seconds": float(
                                    model_pred_time_per_sample.get(spec.name, 0.0)
                                )
                            },
                        }
                    )

                    per_sample_records.append(
                        {
                            "sample_index": sample_idx,
                            "model": f"A::{spec.name}",
                            "metrics": model_metric_values,
                        }
                    )

                    current_pred_i = model_pred_batches[spec.name][i].to(cfg.device)
                    pred_split = _split_current_by_reference(
                        current=current_pred_i,
                        original=original_i,
                        repaired_target_mask=reference["repaired_target_mask"],
                    )
                    pred_segment_metrics = _compute_segmented_metrics(
                        original=original_i,
                        current=current_pred_i,
                        reference=reference,
                        current_split=pred_split,
                        metrics=metrics,
                        density_alpha=args.density_alpha,
                    )
                    segmented_records.append(
                        {
                            "sample_index": sample_idx,
                            "model": f"A::{spec.name}",
                            "metrics": pred_segment_metrics,
                        }
                    )

                    sample_metrics[spec.name] = model_metric_values
                    sample_predictions[spec.name] = model_pred_batches[spec.name][i]
                    sample_segmented[spec.name] = {
                        "current_repaired": pred_split["current_repaired"]
                        .detach()
                        .cpu(),
                        "current_preserved": pred_split["current_preserved"]
                        .detach()
                        .cpu(),
                        "metrics": pred_segment_metrics,
                    }

            if run_scenario_c:
                if gt_fps_i is None:
                    raise RuntimeError("Scenario C requires pipeline FPS ground truth")
                completion_input_t = (
                    torch.from_numpy(pipeline_stages["fps"])
                    .float()
                    .unsqueeze(0)
                    .to(cfg.device)
                )
                for spec, model in model_entries:
                    t_pred0 = time.perf_counter()
                    pred_c = _predict(
                        model,
                        spec.model_type,
                        completion_input_t,
                        target_points=int(gt_fps_i.shape[0]),
                    )[0]
                    pred_c_elapsed = float(time.perf_counter() - t_pred0)

                    per_sample_records.append(
                        {
                            "sample_index": sample_idx,
                            "model": f"T::C::{spec.name}",
                            "metrics": {"seconds": pred_c_elapsed},
                        }
                    )

                    c_metrics = _compute_metric_values_single(
                        pred=pred_c,
                        gt=gt_fps_i,
                        metrics=metrics,
                        density_alpha=args.density_alpha,
                    )
                    per_sample_records.append(
                        {
                            "sample_index": sample_idx,
                            "model": f"C::{spec.name}",
                            "metrics": c_metrics,
                        }
                    )
                    sample_c_predictions[spec.name] = pred_c.detach().cpu()
                    sample_c_metrics[spec.name] = c_metrics

            if run_scenario_a and sample_idx in selected_set:
                selected_payload[sample_idx] = {
                    "category_caption": sample_category_caption,
                    "original": original_i.detach().cpu(),
                    "defected": defected_for_model[i].detach().cpu(),
                    "defected_metrics": defected_metric_values,
                    "predictions": sample_predictions,
                    "metrics": sample_metrics,
                }
                segmented_selected_payload[sample_idx] = {
                    "category_caption": sample_category_caption,
                    "original_preserved": reference["original_preserved"]
                    .detach()
                    .cpu(),
                    "repaired_target": reference["repaired_target"].detach().cpu(),
                    "per_model": sample_segmented,
                }

            if (run_scenario_b or run_scenario_c) and sample_idx in selected_set:
                selected_pipeline_payload[sample_idx] = {
                    "category_caption": sample_category_caption,
                    "gt_full": original_i.detach().cpu(),
                    "gt_fps": gt_fps_i.detach().cpu() if gt_fps_i is not None else None,
                    "stages": {
                        key: torch.from_numpy(value).float()
                        for key, value in pipeline_stages.items()
                    },
                    "stage_metrics": scenario_b_stage_metrics,
                    "stage_timings": dict(pipeline_stage_timings),
                    "c_predictions": sample_c_predictions,
                    "c_metrics": sample_c_metrics,
                }

    main_metric_cols = _metric_columns(per_sample_records, metrics)
    aggregate_rows = _compute_aggregate_table(per_sample_records, main_metric_cols)
    _save_per_sample_csv(metrics_csv, per_sample_records, main_metric_cols)
    _save_aggregate_csv(summary_csv, aggregate_rows)

    segmented_metric_cols = _metric_columns(segmented_records, [])
    segmented_aggregate_rows = _compute_aggregate_table(
        segmented_records,
        segmented_metric_cols,
    )
    _save_per_sample_csv(
        segmented_metrics_csv, segmented_records, segmented_metric_cols
    )
    _save_aggregate_csv(segmented_summary_csv, segmented_aggregate_rows)

    pointclouds = []
    descriptions = []
    badge_labels = []
    badge_details = []
    side_notes = []
    kept_indices: List[int] = []

    for idx in chosen_indices:
        added_any = False

        if run_scenario_a:
            payload = selected_payload.get(idx)
            if payload is not None:
                rows = [payload["original"], payload["defected"]]
                labels = ["Original", "Defected"]
                details = [f"N={rows[0].shape[0]}"]

                defected_lines = [f"N={rows[1].shape[0]}"]
                defected_lines.extend(
                    _metrics_to_lines(payload["defected_metrics"], metrics)
                )
                details.append("\n".join(defected_lines))

                for spec in model_specs:
                    pred = payload["predictions"][spec.name]
                    rows.append(pred)
                    labels.append(spec.name)

                    metric_values = payload["metrics"][spec.name]
                    pred_lines = [f"N={pred.shape[0]}"]
                    pred_lines.extend(_metrics_to_lines(metric_values, metrics))
                    details.append("\n".join(pred_lines))

                pointclouds.append(rows)
                descriptions.append(str(payload.get("category_caption", "")))
                badge_labels.append(labels)
                badge_details.append(details)
                kept_indices.append(idx)
                added_any = True

        if run_scenario_c:
            pp = selected_pipeline_payload.get(idx)
            if pp is not None:
                fps_cloud = pp["stages"].get("fps")
                if fps_cloud is None:
                    continue
                gt_fps = pp.get("gt_fps")
                gt_for_c = gt_fps if gt_fps is not None else pp["gt_full"]

                rows = [gt_for_c, fps_cloud]
                labels = ["GT", "Input"]
                details = [f"N={rows[0].shape[0]}"]

                input_lines = [f"N={fps_cloud.shape[0]}"]
                input_lines.extend(
                    _metrics_to_lines(
                        pp["stage_metrics"].get("fps", {}),
                        metrics,
                    )
                )
                if "fps" in pp["stage_timings"]:
                    input_lines.append(f"T={float(pp['stage_timings']['fps']):.3f}s")
                details.append("\n".join(input_lines))

                if run_scenario_c and pp["c_predictions"]:
                    for spec in model_specs:
                        pred = pp["c_predictions"].get(spec.name)
                        if pred is None:
                            continue
                        rows.append(pred)
                        labels.append(spec.name)
                        pred_lines = [f"N={pred.shape[0]}"]
                        pred_lines.extend(
                            _metrics_to_lines(
                                pp["c_metrics"].get(spec.name, {}), metrics
                            )
                        )
                        details.append("\n".join(pred_lines))

                pointclouds.append(rows)
                descriptions.append(str(pp.get("category_caption", "")))
                badge_labels.append(labels)
                badge_details.append(details)
                kept_indices.append(idx)
                added_any = True

        if not added_any:
            logger.warning(
                "Sample {sample_idx} was selected for gallery but not available after evaluation",
                sample_idx=idx,
            )

    gallery_cfg = GalleryConfig(
        max_sample_cols=args.max_sample_cols,
        views=_parse_views(args.views),
        point_size=args.point_size,
        max_points=args.max_points,
        zoom=args.zoom,
        point_rotation_deg=_parse_xyz_degrees(args.point_rotation),
        dpi=args.dpi,
        badge_fontsize=args.badge_fontsize,
        badge_detail_fontsize=args.metrics_fontsize,
        caption_fontsize=args.caption_fontsize,
        side_note_fontsize=args.side_note_fontsize,
        border_linewidth=args.border_linewidth,
        block_view_width=args.block_view_width,
        block_row_height=args.block_row_height,
        min_figure_width=args.min_figure_width,
        min_figure_height=args.min_figure_height,
        outer_wspace=0,
        outer_hspace=0,
        page_padding=args.page_padding,
        wrap_caption_chars=80,
    )

    phase_gallery_paths: List[Path] = []
    if run_scenario_b and selected_pipeline_payload:
        phase_specs = [
            ("outlier_1", "outlier_before", "Outlier Removal #1"),
            ("denoise", "denoise", "Denoising"),
            ("outlier_2", "outlier_after", "Outlier Removal #2"),
            ("fps", "fps", "FPS"),
        ]

        for stage_key, stage_slug, stage_title in phase_specs:
            phase_pointclouds = []
            phase_descriptions = []
            phase_badge_labels = []
            phase_badge_details = []
            phase_kept_indices: List[int] = []

            for idx in chosen_indices:
                pp = selected_pipeline_payload.get(idx)
                if pp is None:
                    continue

                stages = pp["stages"]
                output_cloud = stages.get(stage_key)
                if output_cloud is None:
                    continue

                if stage_key == "outlier_1":
                    input_key = "input"
                elif stage_key == "denoise":
                    input_key = "outlier_1" if "outlier_1" in stages else "input"
                elif stage_key == "outlier_2":
                    if "denoise" in stages:
                        input_key = "denoise"
                    elif "outlier_1" in stages:
                        input_key = "outlier_1"
                    else:
                        input_key = "input"
                else:  # fps
                    if "outlier_2" in stages:
                        input_key = "outlier_2"
                    elif "denoise" in stages:
                        input_key = "denoise"
                    elif "outlier_1" in stages:
                        input_key = "outlier_1"
                    else:
                        input_key = "input"

                input_cloud = stages.get(input_key)
                if input_cloud is None:
                    continue

                gt_cloud = pp["gt_full"]
                if stage_key == "fps" and pp.get("gt_fps") is not None:
                    gt_cloud = pp["gt_fps"]

                input_metrics = pp["stage_metrics"].get(input_key, {})
                output_metrics = pp["stage_metrics"].get(stage_key, {})

                rows = [gt_cloud, input_cloud, output_cloud]
                labels = ["GT", "Input", "Output"]

                input_lines = [f"N={input_cloud.shape[0]}"]
                input_lines.extend(_metrics_to_lines(input_metrics, metrics))
                if input_key in pp["stage_timings"]:
                    input_lines.append(
                        f"T={float(pp['stage_timings'][input_key]):.3f}s"
                    )

                output_lines = [f"N={output_cloud.shape[0]}"]
                output_lines.extend(_metrics_to_lines(output_metrics, metrics))
                if stage_key in pp["stage_timings"]:
                    output_lines.append(
                        f"T={float(pp['stage_timings'][stage_key]):.3f}s"
                    )

                phase_pointclouds.append(rows)
                phase_descriptions.append(str(pp.get("category_caption", "")))
                phase_badge_labels.append(labels)
                phase_badge_details.append(
                    [
                        f"N={gt_cloud.shape[0]}",
                        "\n".join(input_lines),
                        "\n".join(output_lines),
                    ]
                )
                phase_kept_indices.append(idx)

            if not phase_pointclouds:
                continue

            phase_output_path = gallery_output.with_name(
                f"{gallery_output.stem}_{stage_slug}{gallery_output.suffix}"
            )
            save_dataset_gallery(
                phase_pointclouds,
                str(phase_output_path),
                dataset_name=f"{args.dataset}-evaluation-{stage_title}",
                sample_indices=phase_kept_indices,
                descriptions=phase_descriptions,
                badge_labels=phase_badge_labels,
                badge_details=phase_badge_details,
                side_notes=[],
                config=gallery_cfg,
                seed=args.seed,
            )
            phase_gallery_paths.append(phase_output_path)

    saved_main_gallery = False
    if not pointclouds:
        if run_scenario_b and not (run_scenario_a or run_scenario_c):
            logger.info(
                "Main gallery skipped for scenario B; per-phase galleries were generated instead."
            )
        else:
            logger.warning(
                "No valid samples available for gallery image; skipping gallery save."
            )
    else:

        save_dataset_gallery(
            pointclouds,
            str(gallery_output),
            dataset_name=f"{args.dataset}-evaluation",
            sample_indices=kept_indices,
            descriptions=descriptions,
            badge_labels=badge_labels,
            badge_details=badge_details,
            side_notes=side_notes,
            config=gallery_cfg,
            seed=args.seed,
        )
        saved_main_gallery = True

    segmented_pointclouds = []
    segmented_badge_labels = []
    segmented_badge_details = []
    segmented_descriptions = []
    segmented_kept_indices: List[int] = []

    for idx in chosen_indices:
        payload = segmented_selected_payload.get(idx)
        if payload is None:
            continue

        labels = []
        details = []
        rows = []

        for model_name in [spec.name for spec in model_specs]:
            model_payload = payload["per_model"][model_name]
            model_metrics = model_payload["metrics"]

            repaired_lines = [f"N={model_payload['current_repaired'].shape[0]}"]
            repaired_lines.extend(
                _segmented_metric_lines(
                    model_metrics,
                    prefix="repaired_target_vs_current_repaired",
                    metrics=metrics,
                )
            )
            preserved_lines = [f"N={model_payload['current_preserved'].shape[0]}"]
            preserved_lines.extend(
                _segmented_metric_lines(
                    model_metrics,
                    prefix="original_preserved_vs_current_preserved",
                    metrics=metrics,
                )
            )

            rows.append(payload["repaired_target"])
            labels.append("Target Repaired-Part")
            details.append(f"N={payload['repaired_target'].shape[0]}")

            rows.append(model_payload["current_repaired"])
            labels.append(f"{model_name} Repaired-Part")
            details.append("\n".join(repaired_lines))

            rows.append(payload["original_preserved"])
            labels.append("Target Preserved-Part")
            details.append(f"N={payload['original_preserved'].shape[0]}")

            rows.append(model_payload["current_preserved"])
            labels.append(f"{model_name} Preserved-Part")
            details.append("\n".join(preserved_lines))

        segmented_pointclouds.append(rows)
        segmented_badge_labels.append(labels)
        segmented_badge_details.append(details)
        segmented_descriptions.append(str(payload.get("category_caption", "")))
        segmented_kept_indices.append(idx)

    saved_segmented_gallery = False
    if segmented_pointclouds:
        seg_cfg = gallery_cfg
        save_dataset_gallery(
            segmented_pointclouds,
            str(segmented_gallery_output),
            dataset_name=f"{args.dataset}-segmented-evaluation",
            sample_indices=segmented_kept_indices,
            descriptions=segmented_descriptions,
            badge_labels=segmented_badge_labels,
            badge_details=segmented_badge_details,
            side_notes=[],
            config=seg_cfg,
            seed=args.seed,
        )
        saved_segmented_gallery = True

    if saved_main_gallery:
        logger.info("Saved gallery image to {path}", path=gallery_output)
    else:
        logger.info("Main gallery was not generated for this run.")
    if phase_gallery_paths:
        for phase_path in phase_gallery_paths:
            logger.info("Saved phase gallery image to {path}", path=phase_path)
    logger.info("Saved per-sample metrics to {path}", path=metrics_csv)
    logger.info("Saved aggregate summary to {path}", path=summary_csv)
    if saved_segmented_gallery:
        logger.info(
            "Saved segmented gallery image to {path}", path=segmented_gallery_output
        )
    else:
        logger.info("Segmented gallery was not generated for this run.")
    logger.info(
        "Saved segmented per-sample metrics to {path}", path=segmented_metrics_csv
    )
    logger.info(
        "Saved segmented aggregate summary to {path}", path=segmented_summary_csv
    )


if __name__ == "__main__":
    main()
