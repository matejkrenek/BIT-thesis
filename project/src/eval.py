from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from tqdm import tqdm

from core import (
    ArgSpec,
    ModelConfig,
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
    create_model,
    create_train_val_test_dataloaders,
    load_model_checkpoint,
    logger,
    parse_and_bootstrap,
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


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    out = [v.strip() for v in value.split(",") if v.strip()]
    return out or None


def _parse_indices(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--sample-indices was provided but no indices were parsed")
    return [int(p) for p in parts]


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

        if model_type not in {"pcn", "pointr"}:
            raise ValueError(f"Unsupported model_type '{model_type}' in '{raw}'")
        if not name:
            raise ValueError(f"Model name is empty in '{raw}'")
        if not checkpoint:
            raise ValueError(f"Checkpoint path is empty in '{raw}'")

        specs.append(
            EvalModelSpec(name=name, model_type=model_type, checkpoint=Path(checkpoint))
        )

    return specs


def _default_model_params(model_type: str) -> Dict[str, int]:
    if model_type == "pcn":
        return {
            "num_dense": 16384,
            "latent_dim": 1024,
            "grid_size": 4,
        }
    if model_type == "pointr":
        return {
            "trans_dim": 384,
            "knn_layer": 1,
            "num_pred": 16384,
            "num_query": 224,
        }
    raise ValueError(f"Unsupported model type: {model_type}")


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
            name=spec.model_type, params=_default_model_params(spec.model_type)
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
        elif model_type == "pointr":
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


def _resolve_run_dir(args, default_output_root: Path) -> Path:
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = default_output_root / output_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_eval_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _build_arg_schema() -> List[ArgSpec]:
    return [
        ArgSpec(
            ("--dataset",), {"required": True, "choices": ["shapenet", "modelnet"]}
        ),
        ArgSpec(
            ("--mode",),
            {
                "type": str,
                "default": "basic",
                "choices": ["basic", "advanced"],
                "help": "Choose corruption pipeline used by core dataset builders.",
            },
        ),
        ArgSpec(("--data-root",), {"type": str, "default": None}),
        ArgSpec(
            ("--categories",),
            {"type": str, "default": None, "help": "Comma-separated categories"},
        ),
        ArgSpec(
            ("--model-spec", "--modelspec"),
            {
                "action": "append",
                "dest": "model_specs",
                "default": [],
                "help": "Format: name:model_type:/abs/or/rel/checkpoint.pt (repeatable)",
            },
        ),
        ArgSpec(("--metrics",), {"type": str, "default": "chamfer,hausdorff,dcd"}),
        ArgSpec(("--density-alpha",), {"type": float, "default": 1000.0}),
        ArgSpec(("--batch-size",), {"type": int, "default": 32}),
        ArgSpec(("--num-workers",), {"type": int, "default": 4}),
        ArgSpec(("--num-samples",), {"type": int, "default": 6}),
        ArgSpec(("--sample-indices",), {"type": str, "default": None}),
        ArgSpec(("--seed",), {"type": int, "default": 42}),
        ArgSpec(
            ("--output-dir",),
            {
                "type": str,
                "default": "eval",
                "help": "Output directory for evaluation runs. Relative paths are under OUTPUT_DIR.",
            },
        ),
        ArgSpec(
            ("--run-name",),
            {
                "type": str,
                "default": None,
                "help": "Run subdirectory name inside output-dir (auto-generated when omitted).",
            },
        ),
        ArgSpec(
            ("--gallery-name",), {"type": str, "default": "evaluation_gallery.png"}
        ),
        ArgSpec(
            ("--metrics-csv-name",),
            {"type": str, "default": "evaluation_per_sample.csv"},
        ),
        ArgSpec(
            ("--summary-csv-name",), {"type": str, "default": "evaluation_summary.csv"}
        ),
        ArgSpec(("--views",), {"type": str, "default": "30,45;30,135"}),
        ArgSpec(("--point-size",), {"type": float, "default": 1.5}),
        ArgSpec(("--max-points",), {"type": int, "default": 8192}),
        ArgSpec(("--zoom",), {"type": float, "default": 1.0}),
        ArgSpec(("--dpi",), {"type": int, "default": 300}),
        ArgSpec(("--max-sample-cols",), {"type": int, "default": 2}),
        ArgSpec(("--defect-augmentation-count",), {"type": int, "default": 5}),
        ArgSpec(("--local-dropout-regions",), {"type": int, "default": 5}),
        ArgSpec(("--dense",), {"action": "store_true"}),
        ArgSpec(("--dense-root",), {"type": str, "default": None}),
        ArgSpec(("--dense-num-points",), {"type": int, "default": 100000}),
        ArgSpec(("--normalize",), {"action": "store_true", "default": True}),
        ArgSpec(("--no-normalize",), {"dest": "normalize", "action": "store_false"}),
    ]


def main() -> None:
    args, cfg = parse_and_bootstrap(
        _build_arg_schema(),
        description="Evaluate trained models, save sample gallery, and export aggregate stats.",
        data_subdir="",
    )

    metrics = _parse_metrics(args.metrics)
    model_specs = _parse_model_specs(args.model_specs)

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

    logger.info("Device: {device}", device=cfg.device)
    logger.info("Data root: {root}", root=data_root)
    logger.info("Run output dir: {run_dir}", run_dir=run_dir)

    dataset = _build_dataset(args, data_root)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after loading")

    _, _, test_loader = create_train_val_test_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataset = test_loader.dataset
    logger.info("Test split size: {size}", size=len(test_dataset))

    if args.sample_indices:
        chosen_indices = _parse_indices(args.sample_indices)
        invalid = [i for i in chosen_indices if i < 0 or i >= len(test_dataset)]
        if invalid:
            raise ValueError(
                f"Invalid sample indices {invalid}. Valid range is [0, {len(test_dataset) - 1}]"
            )
        chosen_indices = list(dict.fromkeys(chosen_indices))
    else:
        rng = np.random.default_rng(args.seed)
        k = min(args.num_samples, len(test_dataset))
        chosen_indices = sorted(
            rng.choice(len(test_dataset), size=k, replace=False).tolist()
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

    running_sample_idx = 0
    for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
        originals, padded_defected, defected_lengths = batch
        if originals is None:
            continue

        originals = originals.to(cfg.device, non_blocking=True)
        padded_defected = padded_defected.to(cfg.device, non_blocking=True)
        defected_lengths = defected_lengths.to(cfg.device)

        batch_size_actual = originals.shape[0]
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

        defected_metric_batch = _compute_metric_values_batch(
            defected_for_model,
            originals,
            metrics=metrics,
            density_alpha=args.density_alpha,
            pred_lengths=defected_lengths,
        )

        model_metric_batches: Dict[str, Dict[str, torch.Tensor]] = {}
        model_pred_batches: Dict[str, torch.Tensor] = {}
        for spec, model in model_entries:
            pred = _predict(model, spec.model_type, defected_for_model, target_points)
            model_pred_batches[spec.name] = pred.detach().cpu()
            model_metric_batches[spec.name] = _compute_metric_values_batch(
                pred,
                originals,
                metrics=metrics,
                density_alpha=args.density_alpha,
            )

        for i, sample_idx in enumerate(batch_indices_cpu):
            defected_metric_values = {
                metric_name: float(defected_metric_batch[metric_name][i].item())
                for metric_name in metrics
                if metric_name in defected_metric_batch
            }
            per_sample_records.append(
                {
                    "sample_index": sample_idx,
                    "model": "Defected",
                    "metrics": defected_metric_values,
                }
            )

            sample_predictions: Dict[str, torch.Tensor] = {}
            sample_metrics: Dict[str, Dict[str, float]] = {}
            for spec in model_specs:
                model_metric_values = {
                    metric_name: float(
                        model_metric_batches[spec.name][metric_name][i].item()
                    )
                    for metric_name in metrics
                    if metric_name in model_metric_batches[spec.name]
                }

                per_sample_records.append(
                    {
                        "sample_index": sample_idx,
                        "model": spec.name,
                        "metrics": model_metric_values,
                    }
                )

                sample_metrics[spec.name] = model_metric_values
                sample_predictions[spec.name] = model_pred_batches[spec.name][i]

            if sample_idx in selected_set:
                selected_payload[sample_idx] = {
                    "original": originals[i].detach().cpu(),
                    "defected": defected_for_model[i].detach().cpu(),
                    "defected_metrics": defected_metric_values,
                    "predictions": sample_predictions,
                    "metrics": sample_metrics,
                }

    aggregate_rows = _compute_aggregate_table(per_sample_records, metrics)
    _save_per_sample_csv(metrics_csv, per_sample_records, metrics)
    _save_aggregate_csv(summary_csv, aggregate_rows)

    pointclouds = []
    descriptions = []
    badge_labels = []
    badge_details = []
    side_notes = []
    kept_indices: List[int] = []

    for idx in chosen_indices:
        payload = selected_payload.get(idx)
        if payload is None:
            logger.warning(
                "Sample {sample_idx} was selected for gallery but not available after evaluation",
                sample_idx=idx,
            )
            continue

        rows = [payload["original"], payload["defected"]]
        labels = ["Original", "Defected"]
        details = [f"N={rows[0].shape[0]}"]

        defected_lines = [f"N={rows[1].shape[0]}"]
        defected_lines.extend(_metrics_to_lines(payload["defected_metrics"], metrics))
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
        descriptions.append("")
        badge_labels.append(labels)
        badge_details.append(details)
        kept_indices.append(idx)

    if not pointclouds:
        raise RuntimeError("No valid samples available for gallery image")

    gallery_cfg = GalleryConfig(
        max_sample_cols=args.max_sample_cols,
        views=_parse_views(args.views),
        point_size=args.point_size,
        max_points=args.max_points,
        zoom=args.zoom,
        dpi=args.dpi,
    )

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

    logger.info("Saved gallery image to {path}", path=gallery_output)
    logger.info("Saved per-sample metrics to {path}", path=metrics_csv)
    logger.info("Saved aggregate summary to {path}", path=summary_csv)


if __name__ == "__main__":
    main()
