from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pytorch3d.ops import sample_farthest_points
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core import (
    ModelConfig,
    available_models,
    bootstrap,
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
    create_model,
    load_model_checkpoint,
    logger,
    save_model_checkpoint,
)
from notifications import DiscordNotifier


def _extract_cli_value(argv: list[str], option: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == option:
            if idx + 1 < len(argv):
                return argv[idx + 1]
            return None
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return None


def _parse_gpu_ids(raw: str) -> list[int]:
    ids = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("GPU ids must be >= 0")
        ids.append(value)

    unique_ids = sorted(set(ids))
    if not unique_ids:
        raise ValueError("At least one GPU id must be provided")
    return unique_ids


def _preconfigure_cuda_visible_devices(argv: list[str]) -> list[int] | None:
    raw_gpu_ids = _extract_cli_value(argv, "--gpu-ids")
    raw_num_gpus = _extract_cli_value(argv, "--num-gpus")

    if raw_gpu_ids:
        ids = _parse_gpu_ids(raw_gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in ids)
        return ids

    if raw_num_gpus:
        count = int(raw_num_gpus)
        if count <= 0:
            raise ValueError("--num-gpus must be > 0")
        ids = list(range(count))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in ids)
        return ids

    return None


def _save_loss_plot(
    train_losses: list[float],
    val_full_losses: list[float],
    val_patch_losses: list[float],
    val_combined_losses: list[float],
    path: Path,
    *,
    pointcleannet_mode: bool = False,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="train", linewidth=2)
    if pointcleannet_mode:
        plt.plot(val_patch_losses, label="val", linewidth=2)
    else:
        plt.plot(val_full_losses, label="val_full", linewidth=2)
        plt.plot(val_patch_losses, label="val_patch", linewidth=2)
        plt.plot(val_combined_losses, label="val_combined", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _json_dict(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--model-params must be a JSON object")
    return parsed


def _default_model_params(model_name: str) -> dict[str, Any]:
    if model_name == "pcn":
        return {"num_dense": 16384, "latent_dim": 1024, "grid_size": 4}
    if model_name == "pointr":
        return {
            "trans_dim": 384,
            "knn_layer": 1,
            "num_pred": 16384,
            "num_query": 224,
        }
    if model_name == "adapointr":
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
    if model_name == "pointcleannet":
        return {
            "num_points": 500,
            "num_scales": 1,
            "output_dim": 3,
            "use_point_stn": True,
            "use_feat_stn": True,
            "sym_op": "max",
            "point_tuple": 1,
        }
    if model_name == "pointcleannet_outliers":
        return {
            "num_points": 500,
            "num_scales": 1,
            "output_dim": 1,
            "use_point_stn": True,
            "use_feat_stn": True,
            "sym_op": "max",
            "point_tuple": 1,
        }
    raise ValueError(f"Unsupported model '{model_name}'")


def _default_learning_rate(model_name: str) -> float:
    if model_name == "pcn":
        return 1e-4
    if model_name == "pointr":
        return 5e-5
    if model_name == "adapointr":
        return 5e-5
    if model_name == "pointcleannet":
        return 1e-4
    if model_name == "pointcleannet_outliers":
        return 1e-4
    raise ValueError(f"Unsupported model '{model_name}'")


def _default_weight_decay(model_name: str) -> float:
    if model_name == "pcn":
        return 0.0
    if model_name == "pointr":
        return 1e-4
    if model_name == "adapointr":
        return 1e-4
    if model_name == "pointcleannet":
        return 0.0
    if model_name == "pointcleannet_outliers":
        return 0.0
    raise ValueError(f"Unsupported model '{model_name}'")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _compute_loss(
    model_name: str,
    model: torch.nn.Module,
    prediction,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    core_model = _unwrap_model(model)

    if model_name == "pcn":
        total = core_model.compute_loss(prediction, target)
        return total, {"total": float(total.detach().item())}

    if model_name == "pointr":
        loss_value = core_model.get_loss(prediction, target)
        if isinstance(loss_value, (tuple, list)):
            pieces = [item for item in loss_value if torch.is_tensor(item)]
            if not pieces:
                raise ValueError("PoinTr loss tuple did not contain tensor items")
            total = sum(pieces)
            metrics = {
                "total": float(total.detach().item()),
                "coarse": float(pieces[0].detach().item()),
                "fine": float(pieces[1].detach().item()) if len(pieces) > 1 else 0.0,
            }
            return total, metrics
        total = loss_value
        return total, {"total": float(total.detach().item())}

    if model_name == "adapointr":
        loss_value = core_model.get_loss(prediction, target)
        if isinstance(loss_value, (tuple, list)):
            pieces = [item for item in loss_value if torch.is_tensor(item)]
            if not pieces:
                raise ValueError("AdaPoinTr loss tuple did not contain tensor items")
            total = sum(pieces)
            metrics = {
                "total": float(total.detach().item()),
                "coarse": float(pieces[0].detach().item()),
                "fine": float(pieces[1].detach().item()) if len(pieces) > 1 else 0.0,
            }
            return total, metrics
        total = loss_value
        return total, {"total": float(total.detach().item())}

    if model_name == "pointcleannet":
        total = core_model.get_loss(prediction, target)
        return total, {"total": float(total.detach().item())}

    if model_name == "pointcleannet_outliers":
        total = core_model.get_outlier_loss(prediction, target)
        return total, {"total": float(total.detach().item())}

    raise ValueError(f"Unsupported model '{model_name}'")


def _split_indices(
    n: int, train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _collate_full(batch):
    pairs = []
    for item in batch:
        if item is None:
            continue
        if hasattr(item, "original_pos") and hasattr(item, "defected_pos"):
            pairs.append((item.original_pos, item.defected_pos))
        elif (
            isinstance(item, dict) and "original_pos" in item and "defected_pos" in item
        ):
            pairs.append((item["original_pos"], item["defected_pos"]))

    if not pairs:
        return None, None, None

    originals = []
    defecteds = []
    for original, defected in pairs:
        originals.append(torch.as_tensor(original).float())
        defecteds.append(torch.as_tensor(defected).float())

    originals_t = torch.stack(originals, dim=0)
    lengths = torch.tensor([pc.shape[0] for pc in defecteds], dtype=torch.long)
    max_n = int(lengths.max().item())
    padded = torch.zeros(len(defecteds), max_n, 3, dtype=originals_t.dtype)
    for i, pc in enumerate(defecteds):
        padded[i, : pc.shape[0]] = pc

    return originals_t, padded, lengths


def _collate_patch(batch):
    originals = []
    defecteds = []
    outlier_labels = []

    def _infer_center_outlier_label(
        original_patch: torch.Tensor,
        defected_patch: torch.Tensor,
    ) -> torch.Tensor:
        # Patch is typically centered around the queried point; nearest-to-origin
        # point is used as a robust center proxy.
        if original_patch.numel() == 0 or defected_patch.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        center_idx = int(torch.argmin(torch.norm(defected_patch, dim=1)).item())
        center = defected_patch[center_idx : center_idx + 1]

        center_to_original = torch.cdist(center, original_patch).min()

        if original_patch.shape[0] <= 1:
            threshold = torch.tensor(0.03, device=original_patch.device)
        else:
            d = torch.cdist(original_patch, original_patch)
            d.fill_diagonal_(float("inf"))
            nn = torch.min(d, dim=1).values
            robust_scale = torch.median(nn)
            threshold = torch.clamp(robust_scale * 3.0, min=0.03)

        label = (center_to_original > threshold).float()
        return label.detach().cpu().float()

    for item in batch:
        if item is None:
            continue

        p_orig = torch.as_tensor(item.original_pos).float()
        p_def = torch.as_tensor(item.defected_pos).float()
        if p_orig.ndim != 3 or p_def.ndim != 3:
            continue

        m = int(p_orig.shape[0])
        if m <= 0:
            continue

        patch_idx = int(torch.randint(0, m, (1,)).item())

        o_patch = p_orig[patch_idx]
        d_patch = p_def[patch_idx]

        if hasattr(item, "original_valid_counts") and hasattr(
            item, "defected_valid_counts"
        ):
            o_valid = int(item.original_valid_counts[patch_idx])
            d_valid = int(item.defected_valid_counts[patch_idx])
            o_patch_valid = o_patch[: max(o_valid, 0)]
            d_patch_valid = d_patch[: max(d_valid, 0)]
        else:
            o_patch_valid = o_patch
            d_patch_valid = d_patch

        originals.append(o_patch)
        defecteds.append(d_patch)
        outlier_labels.append(_infer_center_outlier_label(o_patch_valid, d_patch_valid))

    if not originals:
        return None, None, None

    return (
        torch.stack(originals, dim=0),
        torch.stack(defecteds, dim=0),
        torch.stack(outlier_labels, dim=0),
    )


def _next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def _run_mixed_train_epoch(
    *,
    model_name: str,
    model: torch.nn.Module,
    full_loader: DataLoader,
    patch_loader: DataLoader,
    device: torch.device,
    optimizer: AdamW,
    grad_clip: float,
    patch_probability: float,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train(True)

    full_it = iter(full_loader)
    patch_it = iter(patch_loader)

    steps = max(len(full_loader), len(patch_loader))
    total_loss = 0.0
    full_steps = 0
    patch_steps = 0

    batch_progress = tqdm(
        range(steps),
        desc=f"Epoch {epoch}/{total_epochs}",
        unit="batch",
        leave=False,
        position=1,
    )
    for _ in batch_progress:
        use_patch = (
            True
            if model_name in {"pointcleannet", "pointcleannet_outliers"}
            else bool(torch.rand(1).item() < patch_probability)
        )

        if use_patch:
            batch, patch_it = _next_batch(patch_it, patch_loader)
            originals, defected, outlier_labels = batch
            if originals is None or defected is None:
                continue
            originals = originals.to(device, non_blocking=True)
            defected = defected.to(device, non_blocking=True)
            if outlier_labels is not None:
                outlier_labels = outlier_labels.to(device, non_blocking=True)
            patch_steps += 1
        else:
            batch, full_it = _next_batch(full_it, full_loader)
            originals, padded, lengths = batch
            if originals is None or padded is None or lengths is None:
                continue
            originals = originals.to(device, non_blocking=True)
            padded = padded.to(device, non_blocking=True)
            lengths = lengths.to(device)
            defected, _ = sample_farthest_points(
                padded,
                K=originals.shape[1],
                lengths=lengths,
            )
            outlier_labels = None
            full_steps += 1

        prediction = model(defected)
        if model_name == "pointcleannet_outliers":
            if outlier_labels is None:
                continue
            target = outlier_labels
        else:
            target = originals

        loss, metrics = _compute_loss(model_name, model, prediction, target)
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(metrics["total"])
        completed_steps = full_steps + patch_steps
        batch_progress.set_postfix(
            loss=f"{(total_loss / max(completed_steps, 1)):.6f}",
            full=full_steps,
            patch=patch_steps,
        )

    denom = max(full_steps + patch_steps, 1)
    return {
        "loss": total_loss / denom,
        "full_steps": float(full_steps),
        "patch_steps": float(patch_steps),
    }


def _run_eval_full(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.train(False)
    total = 0.0
    n = 0
    with torch.no_grad():
        for originals, padded, lengths in loader:
            if originals is None or padded is None or lengths is None:
                continue
            originals = originals.to(device, non_blocking=True)
            padded = padded.to(device, non_blocking=True)
            lengths = lengths.to(device)
            defected, _ = sample_farthest_points(
                padded,
                K=originals.shape[1],
                lengths=lengths,
            )
            prediction = model(defected)
            loss, metrics = _compute_loss(model_name, model, prediction, originals)
            if not torch.isfinite(loss):
                continue
            total += float(metrics["total"])
            n += 1
    return total / max(n, 1)


def _run_eval_patch(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.train(False)
    total = 0.0
    n = 0
    with torch.no_grad():
        for originals, defected, outlier_labels in loader:
            if originals is None or defected is None:
                continue
            originals = originals.to(device, non_blocking=True)
            defected = defected.to(device, non_blocking=True)
            if outlier_labels is not None:
                outlier_labels = outlier_labels.to(device, non_blocking=True)
            prediction = model(defected)
            if model_name == "pointcleannet_outliers":
                if outlier_labels is None:
                    continue
                target = outlier_labels
            else:
                target = originals
            loss, metrics = _compute_loss(model_name, model, prediction, target)
            if not torch.isfinite(loss):
                continue
            total += float(metrics["total"])
            n += 1
    return total / max(n, 1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune model with mixed full-object and patch batches"
    )
    parser.add_argument("--model", type=str, choices=available_models(), required=True)
    parser.add_argument("--model-params", type=str, default="{}")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        default=False,
        help="When set with --resume-checkpoint, also restore optimizer/scheduler and continue from checkpoint epoch.",
    )
    parser.add_argument("--output-model", type=str, default="best_finetune.pt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overfit", action="store_true", default=False)
    parser.add_argument("--overfit-samples", type=int, default=128)
    parser.add_argument(
        "--discord",
        action="store_true",
        default=False,
        help="Enable Discord training notifications.",
    )
    parser.add_argument(
        "--discord-webhook-url",
        type=str,
        default=None,
        help="Discord webhook URL. If omitted, DISCORD_WEBHOOK_URL env var is used.",
    )
    parser.add_argument(
        "--discord-project-name",
        type=str,
        default="BIT Thesis Project",
        help="Project name displayed in Discord notifications.",
    )
    parser.add_argument(
        "--discord-avatar-name",
        type=str,
        default="FineTune Bot",
        help="Avatar/author name shown in Discord notifications.",
    )
    parser.add_argument(
        "--discord-progress-every",
        type=int,
        default=1,
        help="Send Discord progress update every N epochs.",
    )

    parser.add_argument(
        "--dataset-variant", type=str, choices=["basic", "advanced"], default="basic"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-parallel", action="store_true", default=False)
    parser.add_argument(
        "--auto-data-parallel",
        action="store_true",
        default=True,
        help="Automatically enable DataParallel when multiple GPUs are visible.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use from index 0 (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-step-size", type=int, default=20)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--train-ratio", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--patch-probability", type=float, default=0.5)
    parser.add_argument("--val-patch-weight", type=float, default=0.5)

    parser.add_argument("--patch-size", type=int, default=8192)
    parser.add_argument(
        "--num-patches",
        type=int,
        default=0,
        help="Number of patches per object. Use 0 for automatic selection.",
    )
    parser.add_argument("--normalize-patches", action="store_true")
    parser.add_argument(
        "--patching-method",
        type=str,
        default="pointcleannet_radius",
        choices=["fps_knn", "pointcleannet_radius"],
    )
    parser.add_argument("--patch-radius", type=float, default=0.1)
    parser.add_argument(
        "--patch-center", type=str, default="point", choices=["point", "mean", "none"]
    )
    parser.add_argument("--patch-point-count-std", type=float, default=0.0)

    parser.add_argument("--dense", action="store_true", default=True)
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    preselected_gpu_ids = _preconfigure_cuda_visible_devices(sys.argv[1:])

    parser = _build_parser()
    args = parser.parse_args()

    cfg = bootstrap(seed=int(args.seed))

    if args.num_gpus is not None and args.num_gpus <= 0:
        raise ValueError("--num-gpus must be > 0")

    selected_gpu_ids = preselected_gpu_ids
    if args.gpu_ids:
        selected_gpu_ids = _parse_gpu_ids(args.gpu_ids)
    elif args.num_gpus is not None:
        selected_gpu_ids = list(range(int(args.num_gpus)))

    visible_gpus = torch.cuda.device_count() if cfg.device.type == "cuda" else 0
    auto_data_parallel = bool(args.auto_data_parallel) and visible_gpus > 1
    enable_data_parallel = bool(args.data_parallel) or auto_data_parallel

    if not (0.0 <= args.patch_probability <= 1.0):
        raise ValueError("--patch-probability must be in [0,1]")
    if not (0.0 <= args.val_patch_weight <= 1.0):
        raise ValueError("--val-patch-weight must be in [0,1]")

    model_name = args.model.strip().lower()
    model_params = _default_model_params(model_name)
    model_params.update(_json_dict(args.model_params))

    if model_name in {"pointcleannet", "pointcleannet_outliers"}:
        # PointCleanNet is patch-based by design.
        args.patch_probability = 1.0
        args.val_patch_weight = 1.0
        args.patching_method = "pointcleannet_radius"
        args.patch_center = "point"
        if int(args.num_patches) == 64:
            # Keep backward compatibility with old examples while preventing
            # under-coverage on very large clouds.
            args.num_patches = 0
        required_points = int(model_params.get("num_points", args.patch_size))
        if int(args.patch_size) != required_points:
            logger.warning(
                "Adjusting --patch-size from %d to %d for pointcleannet compatibility",
                int(args.patch_size),
                required_points,
            )
            args.patch_size = required_points

    effective_num_patches = (
        None if int(args.num_patches) <= 0 else int(args.num_patches)
    )

    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else _default_learning_rate(model_name)
    )
    weight_decay = (
        float(args.weight_decay)
        if args.weight_decay is not None
        else _default_weight_decay(model_name)
    )

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else cfg.output_dir.resolve()
    )
    run_name = (
        args.run_name or f"{model_name}_finetune_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoints_dir / args.output_model
    summary_path = run_dir / "training_summary.json"
    loss_curve_path = run_dir / "loss_curve.png"

    dataset_factory = (
        create_advanced_reconstruction_dataset
        if args.dataset_variant == "advanced"
        else create_basic_reconstruction_dataset
    )

    full_dataset = dataset_factory(
        root=str(cfg.data_dir),
        seed=cfg.seed,
    )
    patch_dataset = dataset_factory(
        root=str(cfg.data_dir),
        seed=cfg.seed,
        dense=bool(args.dense),
        dense_root=str(cfg.data_dir / "ShapeNetV2_dense"),
        split_into_patches=True,
        patch_size=int(args.patch_size),
        num_patches=effective_num_patches,
        normalize_patches=bool(args.normalize_patches),
        patching_method=str(args.patching_method),
        patch_radius=float(args.patch_radius),
        patch_center=str(args.patch_center),
        patch_point_count_std=float(args.patch_point_count_std),
        include_full_objects_in_patches=True,
    )

    if effective_num_patches is None:
        logger.info(
            "Using automatic patch count selection (num_patches=None) based on object size"
        )

    if len(full_dataset) != len(patch_dataset):
        raise RuntimeError("Full and patch datasets must have equal length")

    if args.overfit:
        overfit_count = min(max(1, int(args.overfit_samples)), len(full_dataset))
        overfit_indices = list(range(overfit_count))
        full_dataset = Subset(full_dataset, overfit_indices)
        patch_dataset = Subset(patch_dataset, overfit_indices)
        logger.warning(
            f"Overfit mode enabled: using {overfit_count} samples for full+patch datasets"
        )
        train_idx = list(range(len(full_dataset)))
        val_idx = list(range(len(full_dataset)))
        logger.warning(
            "Overfit mode: train and val loaders use the same samples (expected for memorization check)"
        )
    else:
        train_idx, val_idx, _ = _split_indices(
            len(full_dataset),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            seed=cfg.seed,
        )

    pin_memory = cfg.device.type == "cuda"
    common_loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": pin_memory,
    }

    train_full_loader = DataLoader(
        Subset(full_dataset, train_idx),
        shuffle=True,
        collate_fn=_collate_full,
        **common_loader_kwargs,
    )
    val_full_loader = DataLoader(
        Subset(full_dataset, val_idx),
        shuffle=False,
        collate_fn=_collate_full,
        **common_loader_kwargs,
    )
    train_patch_loader = DataLoader(
        Subset(patch_dataset, train_idx),
        shuffle=True,
        collate_fn=_collate_patch,
        **common_loader_kwargs,
    )
    val_patch_loader = DataLoader(
        Subset(patch_dataset, val_idx),
        shuffle=False,
        collate_fn=_collate_patch,
        **common_loader_kwargs,
    )

    model = create_model(
        ModelConfig(name=model_name, params=model_params),
        device=cfg.device,
        data_parallel=enable_data_parallel,
        num_gpus=visible_gpus,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(
        optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma)
    )

    start_epoch = 1
    best_val_combined = float("inf")

    if args.resume_checkpoint:
        if args.resume_training_state:
            loaded = load_model_checkpoint(
                checkpoint_path=Path(args.resume_checkpoint),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=cfg.device,
                strict=True,
                weights_only=False,
            )
            start_epoch = int(loaded.get("epoch", 0)) + 1
            logger.info(
                f"Resumed full training state from {args.resume_checkpoint} (start_epoch={start_epoch})"
            )
        else:
            load_model_checkpoint(
                checkpoint_path=Path(args.resume_checkpoint),
                model=model,
                optimizer=None,
                scheduler=None,
                map_location=cfg.device,
                strict=True,
                weights_only=False,
            )
            start_epoch = 1
            logger.info(
                f"Loaded model weights from {args.resume_checkpoint}; starting finetune from epoch 1"
            )

    if start_epoch > int(args.epochs):
        raise ValueError(
            f"start_epoch ({start_epoch}) is greater than --epochs ({args.epochs}). "
            "Increase --epochs or disable --resume-training-state."
        )

    logger.info(f"Fine-tune model: {model_name}")
    logger.info(f"Device: {cfg.device}")
    if selected_gpu_ids is not None:
        logger.info(f"Requested GPU ids: {selected_gpu_ids}")
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        logger.info(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"Visible CUDA GPUs: {visible_gpus}")
    if auto_data_parallel and not bool(args.data_parallel):
        logger.info("DataParallel enabled automatically (multiple GPUs selected)")
    logger.info(f"Run dir: {run_dir}")
    logger.info(
        f"Train(full/patch)={len(train_full_loader.dataset)}/{len(train_patch_loader.dataset)} | "
        f"Val(full/patch)={len(val_full_loader.dataset)}/{len(val_patch_loader.dataset)}"
    )

    notifier: DiscordNotifier | None = None
    if bool(args.discord):
        webhook_url = args.discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        if webhook_url:
            notifier = DiscordNotifier(
                webhook_url=webhook_url,
                project_name=args.discord_project_name,
                avatar_name=args.discord_avatar_name,
            )
            logger.info("Discord reporting enabled")
        else:
            logger.warning(
                "--discord enabled but no webhook URL provided; Discord reporting is disabled"
            )

    wall_start = time.time()
    train_losses: list[float] = []
    val_full_losses: list[float] = []
    val_patch_losses: list[float] = []
    val_combined_losses: list[float] = []
    if notifier is not None:
        try:
            notifier.send_training_start(
                total_epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                train_size=len(train_full_loader.dataset),
                val_size=len(val_full_loader.dataset),
                training_on=str(cfg.device),
                number_of_gpus=(
                    torch.cuda.device_count() if cfg.device.type == "cuda" else 0
                ),
                learning_rate=learning_rate,
            )
        except Exception as exc:
            logger.warning(f"Discord start notification failed: {exc}")

    progress = tqdm(
        range(start_epoch, int(args.epochs) + 1), desc="FineTune", unit="epoch"
    )
    try:
        for epoch in progress:
            train_stats = _run_mixed_train_epoch(
                model_name=model_name,
                model=model,
                full_loader=train_full_loader,
                patch_loader=train_patch_loader,
                device=cfg.device,
                optimizer=optimizer,
                grad_clip=float(args.grad_clip),
                patch_probability=float(args.patch_probability),
                epoch=epoch,
                total_epochs=int(args.epochs),
            )

            with torch.no_grad():
                if model_name in {"pointcleannet", "pointcleannet_outliers"}:
                    val_full = 0.0
                else:
                    val_full = _run_eval_full(
                        model_name, model, val_full_loader, cfg.device
                    )
                val_patch = _run_eval_patch(
                    model_name, model, val_patch_loader, cfg.device
                )

            val_combined = (1.0 - float(args.val_patch_weight)) * val_full + float(
                args.val_patch_weight
            ) * val_patch
            scheduler.step()

            train_losses.append(float(train_stats["loss"]))
            val_full_losses.append(float(val_full))
            val_patch_losses.append(float(val_patch))
            val_combined_losses.append(float(val_combined))
            _save_loss_plot(
                train_losses,
                val_full_losses,
                val_patch_losses,
                val_combined_losses,
                loss_curve_path,
                pointcleannet_mode=model_name
                in {"pointcleannet", "pointcleannet_outliers"},
            )

            elapsed = time.time() - wall_start
            epochs_done = epoch - start_epoch + 1
            epochs_left = int(args.epochs) - epoch
            avg_epoch_time = elapsed / max(epochs_done, 1)
            eta_seconds = int(avg_epoch_time * max(epochs_left, 0))

            progress.set_postfix(
                train=f"{train_stats['loss']:.6f}",
                val_full=f"{val_full:.6f}",
                val_patch=f"{val_patch:.6f}",
                val_mix=f"{val_combined:.6f}",
                fsteps=int(train_stats["full_steps"]),
                psteps=int(train_stats["patch_steps"]),
                eta=f"{eta_seconds // 60}m {eta_seconds % 60}s",
            )

            metrics = {
                "train_loss": float(train_stats["loss"]),
                "train_full_steps": int(train_stats["full_steps"]),
                "train_patch_steps": int(train_stats["patch_steps"]),
                "val_full_loss": float(val_full),
                "val_patch_loss": float(val_patch),
                "val_combined_loss": float(val_combined),
                "learning_rate": scheduler.get_last_lr()[0],
            }

            if val_combined < best_val_combined:
                best_val_combined = val_combined
                save_model_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    model_config=ModelConfig(name=model_name, params=model_params),
                    extra={"args": vars(args)},
                )
                logger.info(
                    f"Saved best checkpoint at epoch {epoch} with val_combined={val_combined:.6f}"
                )

            if (
                notifier is not None
                and int(args.discord_progress_every) > 0
                and epoch % int(args.discord_progress_every) == 0
            ):
                try:
                    notifier.send_training_progress(
                        epoch=epoch,
                        total_epochs=int(args.epochs),
                        current_loss=val_combined,
                        best_loss=best_val_combined,
                        learning_rate=scheduler.get_last_lr()[0],
                        batch_size=int(args.batch_size),
                        elapsed_time=(
                            f"{int(elapsed // 3600)}h "
                            f"{int((elapsed % 3600) // 60)}m "
                            f"{int(elapsed % 60)}s"
                        ),
                        estimated_finish_time=time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(time.time() + eta_seconds),
                        ),
                        loss_curve_path=loss_curve_path,
                    )
                except Exception as exc:
                    logger.warning(f"Discord progress notification failed: {exc}")
    except Exception as exc:
        if notifier is not None:
            try:
                notifier.send_training_error(
                    error_message=str(exc),
                    epoch=epoch if "epoch" in locals() else None,
                )
            except Exception as notify_exc:
                logger.warning(f"Discord error notification failed: {notify_exc}")
        raise

    summary = {
        "model": model_name,
        "model_params": model_params,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_val_combined": best_val_combined,
        "patch_probability": float(args.patch_probability),
        "val_patch_weight": float(args.val_patch_weight),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "loss_curve": str(loss_curve_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Fine-tune finished. Best mixed val: {best_val_combined:.6f}")
    logger.info(f"Best checkpoint: {best_checkpoint_path}")

    if notifier is not None:
        try:
            total_seconds = int(time.time() - wall_start)
            notifier.send_training_completion(
                total_epochs=int(args.epochs),
                final_loss=float(best_val_combined),
                best_loss=float(best_val_combined),
                training_time=(
                    f"{total_seconds // 3600}h "
                    f"{(total_seconds % 3600) // 60}m "
                    f"{total_seconds % 60}s"
                ),
                final_loss_curve_path=loss_curve_path,
                best_model_path=best_checkpoint_path,
            )
        except Exception as exc:
            logger.warning(f"Discord completion notification failed: {exc}")


if __name__ == "__main__":
    main()
