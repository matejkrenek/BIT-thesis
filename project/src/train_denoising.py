from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core import (
    bootstrap,
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
    load_model_checkpoint,
    logger,
    save_model_checkpoint,
)
from models.pointcleannet import PointCleanNetHybrid
from notifications import DiscordNotifier


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PointCleanNet hybrid on patch pairs (denoise + outlier)"
    )

    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--resume-training-state", action="store_true", default=False)
    parser.add_argument("--output-model", type=str, default="best_denoising.pt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument(
        "--dataset-variant", type=str, choices=["basic", "advanced"], default="basic"
    )
    parser.add_argument("--dense", action="store_true", default=True)
    parser.add_argument("--no-dense", dest="dense", action="store_false")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--patch-size", type=int, default=500)
    parser.add_argument("--num-patches", type=int, default=128)
    parser.add_argument("--normalize-patches", action="store_true")
    parser.add_argument(
        "--patching-method",
        type=str,
        default="pointcleannet_radius",
        choices=["fps_knn", "pointcleannet_radius"],
    )
    parser.add_argument("--patch-radius", type=float, default=0.05)
    parser.add_argument(
        "--patch-center", type=str, default="point", choices=["point", "mean", "none"]
    )
    parser.add_argument("--patch-point-count-std", type=float, default=0.0)

    parser.add_argument("--surface-loss-alpha", type=float, default=0.99)
    parser.add_argument("--denoise-loss-weight", type=float, default=1.0)
    parser.add_argument("--outlier-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--outlier-target-distance",
        type=float,
        default=-1.0,
        help="Distance threshold for pseudo outlier labels. If < 0, uses max(0.02, patch_radius * 1.5).",
    )
    parser.add_argument("--outlier-threshold", type=float, default=0.5)
    parser.add_argument("--min-kept-points", type=int, default=32)

    parser.add_argument("--use-point-stn", action="store_true", default=True)
    parser.add_argument(
        "--no-use-point-stn", dest="use_point_stn", action="store_false"
    )
    parser.add_argument("--use-feat-stn", action="store_true", default=True)
    parser.add_argument("--no-use-feat-stn", dest="use_feat_stn", action="store_false")
    parser.add_argument("--sym-op", type=str, default="max")
    parser.add_argument("--point-tuple", type=int, default=1)

    parser.add_argument("--overfit", action="store_true", default=False)
    parser.add_argument("--overfit-samples", type=int, default=128)

    parser.add_argument("--discord", action="store_true", default=False)
    parser.add_argument("--discord-webhook-url", type=str, default=None)
    parser.add_argument(
        "--discord-project-name", type=str, default="BIT Thesis Project"
    )
    parser.add_argument("--discord-avatar-name", type=str, default="Denoising Bot")
    parser.add_argument("--discord-progress-every", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    return parser


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
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def _collate_patch(batch):
    originals = []
    defecteds = []
    patch_centers = []
    original_fulls = []

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

        centers = getattr(item, "patch_centers", None)
        original_full = getattr(item, "original_full_pos", None)
        if centers is None or original_full is None:
            continue

        centers = torch.as_tensor(centers).float()
        original_full = torch.as_tensor(original_full).float()
        if centers.ndim != 2 or centers.shape[1] != 3:
            continue
        if original_full.ndim != 2 or original_full.shape[1] != 3:
            continue

        patch_idx = int(torch.randint(0, m, (1,)).item())
        originals.append(p_orig[patch_idx])
        defecteds.append(p_def[patch_idx])
        patch_centers.append(centers[patch_idx])
        original_fulls.append(original_full)

    if not originals:
        return None, None, None, None

    return (
        torch.stack(originals, dim=0),
        torch.stack(defecteds, dim=0),
        torch.stack(patch_centers, dim=0),
        original_fulls,
    )


def _surface_distance_loss(
    pred_points: torch.Tensor, gt_patch_points: torch.Tensor, alpha: float = 0.99
) -> torch.Tensor:
    # pred_points: (B, 3), gt_patch_points: (B, K, 3)
    dists = torch.cdist(pred_points.unsqueeze(1), gt_patch_points).squeeze(1)  # (B, K)
    min_dist = dists.min(dim=1).values
    max_dist = dists.max(dim=1).values
    return torch.mean(alpha * min_dist + (1.0 - alpha) * max_dist) * 100.0


def _save_loss_plot(
    train_losses: list[float], val_losses: list[float], path: Path
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train", linewidth=2)
    plt.plot(val_losses, label="val", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _compute_outlier_targets(
    patch_centers: torch.Tensor,
    original_fulls: list[torch.Tensor],
    device: torch.device,
    threshold: float,
) -> torch.Tensor:
    targets = []
    for center, clean_cloud in zip(patch_centers, original_fulls):
        clean_cloud = clean_cloud.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        center = center.to(device=device, dtype=torch.float32, non_blocking=True)
        nn_dist = torch.cdist(
            center.unsqueeze(0).unsqueeze(0), clean_cloud.unsqueeze(0)
        ).squeeze()
        min_dist = float(nn_dist.min().item())
        targets.append(1.0 if min_dist > threshold else 0.0)
    return torch.tensor(targets, device=device, dtype=torch.float32)


def _run_epoch(
    *,
    model: PointCleanNetHybrid,
    loader: DataLoader,
    device: torch.device,
    training: bool,
    optimizer: AdamW | None,
    grad_clip: float,
    surface_loss_alpha: float,
    outlier_target_distance: float,
    denoise_loss_weight: float,
    outlier_loss_weight: float,
) -> tuple[float, float, float]:
    if training and optimizer is None:
        raise ValueError("optimizer is required for training")

    model.train(training)

    total = 0.0
    total_denoise = 0.0
    total_outlier = 0.0
    n = 0

    for originals, defecteds, patch_centers, original_fulls in loader:
        if originals is None or defecteds is None:
            continue

        originals = originals.to(device, non_blocking=True)
        defecteds = defecteds.to(device, non_blocking=True)
        patch_centers = patch_centers.to(device, non_blocking=True)

        outputs = model(defecteds)
        if not isinstance(outputs, dict):
            raise TypeError(
                "Expected PointCleanNetHybrid to return a dict for (P,K,3) input"
            )

        pred_points = outputs["denoise_displacements"]
        outlier_scores = outputs["outlier_scores"]

        outlier_targets = _compute_outlier_targets(
            patch_centers=patch_centers,
            original_fulls=original_fulls,
            device=device,
            threshold=outlier_target_distance,
        )

        denoise_loss = _surface_distance_loss(
            pred_points, originals, alpha=surface_loss_alpha
        )
        outlier_loss = F.l1_loss(outlier_scores, outlier_targets)
        loss = denoise_loss_weight * denoise_loss + outlier_loss_weight * outlier_loss

        if not torch.isfinite(loss):
            continue

        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total += float(loss.detach().item())
        total_denoise += float(denoise_loss.detach().item())
        total_outlier += float(outlier_loss.detach().item())
        n += 1

    n_safe = max(n, 1)
    return total / n_safe, total_denoise / n_safe, total_outlier / n_safe


def main() -> None:
    args = _build_parser().parse_args()
    cfg = bootstrap(seed=int(args.seed))

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else cfg.output_dir.resolve()
    )
    run_name = (
        args.run_name or f"pointcleannet_denoising_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_root / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = ckpt_dir / args.output_model
    summary_path = run_dir / "training_summary.json"
    loss_curve_path = run_dir / "loss_curve.png"

    dataset_factory = (
        create_advanced_reconstruction_dataset
        if args.dataset_variant == "advanced"
        else create_basic_reconstruction_dataset
    )
    patch_dataset = dataset_factory(
        root=str(cfg.data_dir),
        seed=cfg.seed,
        dense=bool(args.dense),
        dense_root=str(cfg.data_dir / "ShapeNetV2_dense"),
        split_into_patches=True,
        patch_size=int(args.patch_size),
        num_patches=int(args.num_patches),
        normalize_patches=bool(args.normalize_patches),
        patching_method=str(args.patching_method),
        patch_radius=float(args.patch_radius),
        patch_center=str(args.patch_center),
        patch_point_count_std=float(args.patch_point_count_std),
        include_full_objects_in_patches=True,
    )

    if args.overfit:
        overfit_count = min(max(1, int(args.overfit_samples)), len(patch_dataset))
        patch_dataset = Subset(patch_dataset, list(range(overfit_count)))
        logger.warning(f"Overfit mode enabled: using {overfit_count} patch samples")

    train_idx, val_idx, _ = _split_indices(
        len(patch_dataset),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=cfg.seed,
    )

    pin_memory = cfg.device.type == "cuda"
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": pin_memory,
        "collate_fn": _collate_patch,
    }

    train_loader = DataLoader(
        Subset(patch_dataset, train_idx), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(patch_dataset, val_idx), shuffle=False, **loader_kwargs
    )

    model = PointCleanNetHybrid(
        outlier_checkpoint_file=None,
        denoise_checkpoint_file=None,
        points_per_patch=int(args.patch_size),
        outlier_threshold=float(args.outlier_threshold),
        min_kept_points=int(args.min_kept_points),
    ).to(cfg.device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = StepLR(
        optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma)
    )

    start_epoch = 1
    best_val_loss = float("inf")

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
                f"Loaded model weights from {args.resume_checkpoint}; starting training from epoch 1"
            )

    if start_epoch > int(args.epochs):
        raise ValueError(
            f"start_epoch ({start_epoch}) is greater than --epochs ({args.epochs}). "
            "Increase --epochs or disable --resume-training-state."
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

    logger.info("Training PointCleanNet hybrid model (denoise + outlier)")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(
        f"Train size={len(train_loader.dataset)}, Val size={len(val_loader.dataset)}"
    )
    logger.info("Model backend: PointCleanNetHybrid (joint branch supervision)")

    outlier_target_distance = float(args.outlier_target_distance)
    if outlier_target_distance < 0.0:
        outlier_target_distance = max(0.02, float(args.patch_radius) * 1.5)
    logger.info(f"Outlier pseudo-label threshold: {outlier_target_distance:.6f}")

    train_losses: list[float] = []
    val_losses: list[float] = []
    wall_start = time.time()

    if notifier is not None:
        try:
            notifier.send_training_start(
                total_epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                train_size=len(train_loader.dataset),
                val_size=len(val_loader.dataset),
                training_on=str(cfg.device),
                number_of_gpus=(
                    torch.cuda.device_count() if cfg.device.type == "cuda" else 0
                ),
                learning_rate=float(args.learning_rate),
            )
        except Exception as exc:
            logger.warning(f"Discord start notification failed: {exc}")

    progress = tqdm(
        range(start_epoch, int(args.epochs) + 1), desc="TrainDenoising", unit="epoch"
    )
    try:
        for epoch in progress:
            train_loss, train_denoise_loss, train_outlier_loss = _run_epoch(
                model=model,
                loader=train_loader,
                device=cfg.device,
                training=True,
                optimizer=optimizer,
                grad_clip=float(args.grad_clip),
                surface_loss_alpha=float(args.surface_loss_alpha),
                outlier_target_distance=outlier_target_distance,
                denoise_loss_weight=float(args.denoise_loss_weight),
                outlier_loss_weight=float(args.outlier_loss_weight),
            )

            with torch.no_grad():
                val_loss, val_denoise_loss, val_outlier_loss = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=cfg.device,
                    training=False,
                    optimizer=None,
                    grad_clip=float(args.grad_clip),
                    surface_loss_alpha=float(args.surface_loss_alpha),
                    outlier_target_distance=outlier_target_distance,
                    denoise_loss_weight=float(args.denoise_loss_weight),
                    outlier_loss_weight=float(args.outlier_loss_weight),
                )

            scheduler.step()

            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            _save_loss_plot(train_losses, val_losses, loss_curve_path)

            elapsed = time.time() - wall_start
            epochs_done = epoch - start_epoch + 1
            epochs_left = int(args.epochs) - epoch
            avg_epoch_time = elapsed / max(epochs_done, 1)
            eta_seconds = int(avg_epoch_time * max(epochs_left, 0))

            progress.set_postfix(
                train=f"{train_loss:.6f}",
                val=f"{val_loss:.6f}",
                train_d=f"{train_denoise_loss:.4f}",
                train_o=f"{train_outlier_loss:.4f}",
                best=f"{best_val_loss:.6f}",
                eta=f"{eta_seconds // 60}m {eta_seconds % 60}s",
            )

            metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_denoise_loss": float(train_denoise_loss),
                "train_outlier_loss": float(train_outlier_loss),
                "val_denoise_loss": float(val_denoise_loss),
                "val_outlier_loss": float(val_outlier_loss),
                "learning_rate": scheduler.get_last_lr()[0],
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    model_config={
                        "name": "pointcleannethybrid",
                        "params": {
                            "points_per_patch": int(args.patch_size),
                            "outlier_checkpoint_file": None,
                            "denoise_checkpoint_file": None,
                        },
                    },
                    extra={"args": vars(args)},
                )
                logger.info(
                    f"Saved best checkpoint at epoch {epoch} with val_loss={val_loss:.6f}"
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
                        current_loss=val_loss,
                        best_loss=best_val_loss,
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
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "best_val_loss": float(best_val_loss),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "loss_curve": str(loss_curve_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Training finished. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Best checkpoint: {best_checkpoint_path}")

    if notifier is not None:
        try:
            total_seconds = int(time.time() - wall_start)
            notifier.send_training_completion(
                total_epochs=int(args.epochs),
                final_loss=val_losses[-1] if val_losses else float("nan"),
                best_loss=best_val_loss,
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
