import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from pytorch3d.ops import sample_farthest_points
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import AugmentedDataset, ModelNetDataset, ShapeNetDataset
from dataset.defect import Combined, LargeMissingRegion, LocalDropout, Noise
from metrics import chamfer_distance_metric, fscore_metric, hausdorff_distance_metric
from models import PCN
from visualize.dataset_gallery import GalleryConfig, save_dataset_gallery


@dataclass
class ModelSpec:
    name: str
    model_type: str
    checkpoint: str


SUPPORTED_METRICS = ("chamfer", "hausdorff", "fscore")

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


def _parse_model_specs(values: Sequence[str]) -> List[ModelSpec]:
    if not values:
        raise ValueError("At least one --model-spec must be provided")

    specs: List[ModelSpec] = []
    for raw in values:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Invalid --model-spec format. Expected name:model_type:/path/to/checkpoint"
            )

        name = parts[0].strip()
        model_type = parts[1].strip().lower()
        checkpoint = parts[2].strip()

        if model_type not in {"pcn"}:
            raise ValueError(f"Unsupported model_type '{model_type}' in '{raw}'")
        if not name:
            raise ValueError(f"Model name is empty in '{raw}'")
        if not checkpoint:
            raise ValueError(f"Checkpoint path is empty in '{raw}'")

        specs.append(ModelSpec(name=name, model_type=model_type, checkpoint=checkpoint))
    return specs


def _load_state_dict(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
        print(f"[WARN] Missing keys for model load ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys for model load ({len(unexpected)}): {unexpected[:5]}")


def _build_model(spec: ModelSpec, device: str) -> torch.nn.Module:
    if spec.model_type == "pcn":
        model = PCN(num_dense=16384, latent_dim=1024, grid_size=4)
    else:
        raise ValueError(f"Unsupported model type: {spec.model_type}")

    _load_state_dict(model, spec.checkpoint, device)
    model = model.to(device)
    model.eval()
    return model


def _predict(model: torch.nn.Module, model_type: str, defected_batched: torch.Tensor, target_points: int) -> torch.Tensor:
    with torch.no_grad():
        if model_type == "pcn":
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
    fscore_threshold: float,
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
    if "fscore" in metrics:
        out["fscore"] = fscore_metric(
            pred,
            gt,
            threshold=fscore_threshold,
            pred_lengths=pred_lengths,
            gt_lengths=gt_lengths,
            reduction="none",
        )

    return out


def _batch_collate(batch):
    batch = [b for b in batch if b is not None and b[1] is not None]
    if len(batch) == 0:
        return None

    originals, defecteds = zip(*batch)

    originals = torch.stack(originals, dim=0)

    lengths = torch.tensor([pc.shape[0] for pc in defecteds], dtype=torch.long)
    max_n = lengths.max().item()

    padded = torch.zeros(len(defecteds), max_n, 3)
    for i, pc in enumerate(defecteds):
        padded[i, : pc.shape[0]] = pc

    return originals, padded, lengths


def _build_dataset_loader(args: argparse.Namespace) -> Tuple[Dataset, DataLoader]:
    categories = _parse_csv(args.categories)

    if args.dataset == "shapenet":
        base = ShapeNetDataset(
            root=os.path.join(args.data_root, "ShapeNetV2"),
            categories=categories,
        )
    elif args.dataset == "modelnet":
        base = ModelNetDataset(
            root=os.path.join(args.data_root, "ModelNet40"),
            categories=categories,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    rng = np.random.default_rng(args.seed)

    defects = [
        Combined(
            [
                LargeMissingRegion(removal_fraction=rng.uniform(0.3, 0.5)),
                LocalDropout(
                    radius=rng.uniform(0.01, 0.1),
                    regions=5,
                    dropout_rate=rng.uniform(0.5, 0.9),
                ),
                Noise(rng.uniform(0.001, 0.005)),
            ]
        )
        for _ in range(5)
    ]

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset = AugmentedDataset(dataset=base, defects=defects)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=g)


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        collate_fn=_batch_collate,
        pin_memory=True,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_batch_collate,
        num_workers=args.num_workers,
        generator=g,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_batch_collate,
        num_workers=args.num_workers,
        generator=g,
    )
    return train_loader, val_loader, test_loader


def _format_metric_short(name: str, value: float) -> str:
    if name == "fscore":
        return f"F0.5={value:.4f}"
    if name == "chamfer":
        return f"CD={value:.5f}"
    if name == "hausdorff":
        return f"HD={value:.5f}"
    return f"{name}={value:.5f}"


def _metrics_to_lines(metric_values: Dict[str, float], metrics: Sequence[str]) -> List[str]:
    return [_format_metric_short(m, metric_values[m]) for m in metrics if m in metric_values]


def _compute_aggregate_table(records: List[Dict[str, object]], metrics: Sequence[str]) -> List[Dict[str, object]]:
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for rec in records:
        model_name = str(rec["model"])
        metric_values = rec["metrics"]
        if model_name not in by_model:
            by_model[model_name] = {m: [] for m in metrics}
        for m in metrics:
            if m in metric_values:
                by_model[model_name][m].append(float(metric_values[m]))

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


def _save_per_sample_csv(path: str, records: List[Dict[str, object]], metrics: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = ["sample_index", "model"] + list(metrics)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rec in records:
            row = [rec["sample_index"], rec["model"]]
            metric_values = rec["metrics"]
            for m in metrics:
                row.append(metric_values.get(m, ""))
            writer.writerow(row)


def _save_aggregate_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
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


def _print_aggregate_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("[WARN] No aggregate rows to print")
        return

    print("\n===== AGGREGATED METRICS (WHOLE DATASET) =====")
    print(f"{'Model':20s} {'Metric':10s} {'Mean':>12s} {'Median':>12s} {'Std':>12s} {'N':>8s}")
    for row in rows:
        print(
            f"{str(row['model']):20.20s} {str(row['metric']):10.10s} "
            f"{float(row['mean']):12.6f} {float(row['median']):12.6f} {float(row['std']):12.6f} {int(row['count']):8d}"
        )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate trained models, save sample gallery with metrics, and export aggregate stats.",
    )

    parser.add_argument("--dataset", required=True, choices=["shapenet", "modelnet"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated categories")

    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Format: name:model_type:/abs/or/rel/checkpoint.pt (repeatable)",
    )

    parser.add_argument("--metrics", type=str, default="chamfer,hausdorff,fscore")
    parser.add_argument("--fscore-threshold", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gallery-output", type=str, default="evaluation_gallery.png")
    parser.add_argument("--metrics-csv", type=str, default="evaluation_per_sample.csv")
    parser.add_argument("--summary-csv", type=str, default="evaluation_summary.csv")

    parser.add_argument("--views", type=str, default="30,45;30,135")
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-sample-cols", type=int, default=2)

    args = parser.parse_args()

    data_folder_path = os.getenv("DATA_FOLDER_PATH", "")
    args.data_root = args.data_root or os.path.join(data_folder_path, "data")

    metrics = _parse_metrics(args.metrics)
    model_specs = _parse_model_specs(args.model_spec)

    loader, _, _ = _build_dataset_loader(args)
    dataset = loader.dataset

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after loading")

    print(f"[INFO] Dataset size: {len(dataset)}")

    if args.sample_indices:
        chosen_indices = _parse_indices(args.sample_indices)
        invalid = [i for i in chosen_indices if i < 0 or i >= len(dataset)]
        if invalid:
            raise ValueError(
                f"Invalid sample indices {invalid}. Valid range is [0, {len(dataset) - 1}]"
            )
        chosen_indices = list(dict.fromkeys(chosen_indices))
    else:
        rng = np.random.default_rng(args.seed)
        k = min(args.num_samples, len(dataset))
        chosen_indices = sorted(rng.choice(len(dataset), size=k, replace=False).tolist())

    selected_set = set(chosen_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model_entries: List[Tuple[ModelSpec, torch.nn.Module]] = []
    for spec in model_specs:
        print(f"[INFO] Loading model '{spec.name}' ({spec.model_type}) from {spec.checkpoint}")
        model_entries.append((spec, _build_model(spec, device)))

    per_sample_records: List[Dict[str, object]] = []
    selected_payload: Dict[int, Dict[str, object]] = {}

    running_sample_idx = 0
    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        if batch is None:
            continue

        originals, padded_defected, defected_lengths = batch

        originals = originals.to(device, non_blocking=True)
        padded_defected = padded_defected.to(device, non_blocking=True)
        defected_lengths = defected_lengths.to(device)

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
            fscore_threshold=args.fscore_threshold,
            pred_lengths=defected_lengths,
        )

        model_metric_batches: Dict[str, Dict[str, torch.Tensor]] = {}
        model_pred_batches: Dict[str, torch.Tensor] = {}
        for spec, model in model_entries:
            pred = _predict(model, spec.model_type, defected_for_model, target_points=target_points)
            model_pred_batches[spec.name] = pred.detach().cpu()
            model_metric_batches[spec.name] = _compute_metric_values_batch(
                pred,
                originals,
                metrics=metrics,
                fscore_threshold=args.fscore_threshold,
            )

        for i, sample_idx in enumerate(batch_indices_cpu):
            defected_metric_values = {
                m: float(defected_metric_batch[m][i].item()) for m in metrics if m in defected_metric_batch
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
                    m: float(model_metric_batches[spec.name][m][i].item())
                    for m in metrics
                    if m in model_metric_batches[spec.name]
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

    _save_per_sample_csv(args.metrics_csv, per_sample_records, metrics)
    _save_aggregate_csv(args.summary_csv, aggregate_rows)
    _print_aggregate_table(aggregate_rows)

    pointclouds = []
    descriptions = []
    badge_labels = []
    badge_details = []
    side_notes = []
    kept_indices: List[int] = []

    for idx in chosen_indices:
        payload = selected_payload.get(idx)
        if payload is None:
            print(f"[WARN] Sample {idx} was selected for gallery but not available after evaluation")
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

    cfg = GalleryConfig(
        max_sample_cols=args.max_sample_cols,
        views=_parse_views(args.views),
        point_size=args.point_size,
        max_points=args.max_points,
        zoom=args.zoom,
        dpi=args.dpi,
    )

    save_dataset_gallery(
        pointclouds,
        args.gallery_output,
        dataset_name=f"{args.dataset}-evaluation",
        sample_indices=kept_indices,
        descriptions=descriptions,
        badge_labels=badge_labels,
        badge_details=badge_details,
        side_notes=side_notes,
        config=cfg,
        seed=args.seed,
    )

    print(f"\n[INFO] Saved gallery image to {args.gallery_output}")
    print(f"[INFO] Saved per-sample metrics to {args.metrics_csv}")
    print(f"[INFO] Saved aggregate summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
