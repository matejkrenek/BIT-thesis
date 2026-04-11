from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from pytorch3d.ops import sample_farthest_points
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core import (
    ArgSpec,
    ModelConfig,
    available_models,
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
    create_model,
    create_train_val_test_dataloaders,
    load_model_checkpoint,
    logger,
    parse_and_bootstrap,
    save_model_checkpoint,
)
from notifications import DiscordNotifier


def _json_dict(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --model-params: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--model-params must be a JSON object")

    return parsed


def _default_model_params(model_name: str) -> dict[str, Any]:
    if model_name == "pcn":
        return {
            "num_dense": 16384,
            "latent_dim": 1024,
            "grid_size": 4,
        }
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
    raise ValueError(f"Unsupported model '{model_name}'")


def _default_learning_rate(model_name: str) -> float:
    if model_name == "pcn":
        return 1e-3
    if model_name == "pointr":
        return 3e-4
    if model_name == "adapointr":
        return 3e-4
    raise ValueError(f"Unsupported model '{model_name}'")


def _default_weight_decay(model_name: str) -> float:
    if model_name == "pcn":
        return 0.0
    if model_name == "pointr":
        return 1e-4
    if model_name == "adapointr":
        return 1e-4
    raise ValueError(f"Unsupported model '{model_name}'")


def _cap_batch_size_for_model(model_name: str, batch_size: int, overfit: bool) -> int:
    if model_name == "adapointr":
        # AdaPoinTr decoder-attention is memory heavy on 8GB GPUs.
        return min(batch_size, 2 if overfit else 4)
    return batch_size


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

    raise ValueError(f"Unsupported model '{model_name}'")


def _run_epoch(
    *,
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    training: bool,
    optimizer: AdamW | None = None,
    grad_clip: float | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[float, dict[str, float]]:
    if training and optimizer is None:
        raise ValueError("optimizer is required when training=True")

    model.train(mode=training)

    total_loss = 0.0
    total_coarse = 0.0
    total_fine = 0.0
    batches = 0

    desc = f"Epoch {epoch}/{total_epochs}" if (training and epoch is not None and total_epochs is not None) else None
    batch_iter = tqdm(loader, desc=desc, unit="batch", leave=False, position=1) if training and desc else loader

    for originals, padded, lengths in batch_iter:

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

        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += float(metrics["total"])
        total_coarse += float(metrics.get("coarse", 0.0))
        total_fine += float(metrics.get("fine", 0.0))
        batches += 1

        if training and hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(loss=f"{total_loss / max(batches, 1):.6f}")

    denom = max(batches, 1)
    avg_loss = total_loss / denom
    avg_metrics = {
        "total": avg_loss,
        "coarse": total_coarse / denom,
        "fine": total_fine / denom,
    }
    return avg_loss, avg_metrics


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


def _build_schema() -> list[ArgSpec]:
    return [
        ArgSpec(
            flags=("--model",),
            kwargs={
                "type": str,
                "choices": available_models(),
                "required": True,
                "help": "Model to train.",
            },
        ),
        ArgSpec(
            flags=("--model-params",),
            kwargs={
                "type": str,
                "default": "{}",
                "help": "JSON object with model constructor params.",
            },
        ),
        ArgSpec(
            flags=("--resume-checkpoint",),
            kwargs={
                "type": str,
                "default": None,
                "help": "Path to a checkpoint for resuming training.",
            },
        ),
        ArgSpec(
            flags=("--output-model",),
            kwargs={
                "type": str,
                "default": "best.pt",
                "help": "File name for the best model checkpoint.",
            },
        ),
        ArgSpec(
            flags=("--save-checkpoints",),
            kwargs={
                "dest": "save_checkpoints",
                "action": "store_true",
                "default": True,
                "help": "Enable periodic checkpoint saves.",
            },
        ),
        ArgSpec(
            flags=("--no-save-checkpoints",),
            kwargs={
                "dest": "save_checkpoints",
                "action": "store_false",
                "help": "Disable periodic checkpoint saves.",
            },
        ),
        ArgSpec(
            flags=("--save-every",),
            kwargs={
                "type": int,
                "default": 10,
                "help": "Save periodic checkpoint every N epochs.",
            },
        ),
        ArgSpec(
            flags=("--overfit",),
            kwargs={
                "action": "store_true",
                "default": False,
                "help": "Use only a small subset for overfit debugging.",
            },
        ),
        ArgSpec(
            flags=("--overfit-samples",),
            kwargs={
                "type": int,
                "default": 128,
                "help": "Number of samples used when --overfit is set.",
            },
        ),
        ArgSpec(
            flags=("--output-dir",),
            kwargs={
                "type": str,
                "default": None,
                "help": "Root output directory for this run.",
            },
        ),
        ArgSpec(
            flags=("--run-name",),
            kwargs={
                "type": str,
                "default": None,
                "help": "Name of output run directory. Auto-generated if omitted.",
            },
        ),
        ArgSpec(
            flags=("--epochs",),
            kwargs={"type": int, "default": 100},
        ),
        ArgSpec(
            flags=("--batch-size",),
            kwargs={"type": int, "default": 64},
        ),
        ArgSpec(
            flags=("--learning-rate",),
            kwargs={"type": float, "default": None},
        ),
        ArgSpec(
            flags=("--weight-decay",),
            kwargs={"type": float, "default": None},
        ),
        ArgSpec(
            flags=("--lr-step-size",),
            kwargs={"type": int, "default": 50},
        ),
        ArgSpec(
            flags=("--lr-gamma",),
            kwargs={"type": float, "default": 0.5},
        ),
        ArgSpec(
            flags=("--grad-clip",),
            kwargs={"type": float, "default": 1.0},
        ),
        ArgSpec(
            flags=("--train-ratio",),
            kwargs={"type": float, "default": 0.8},
        ),
        ArgSpec(
            flags=("--val-ratio",),
            kwargs={"type": float, "default": 0.1},
        ),
        ArgSpec(
            flags=("--num-workers",),
            kwargs={"type": int, "default": 4},
        ),
        ArgSpec(
            flags=("--dataset-variant",),
            kwargs={
                "type": str,
                "choices": ["basic", "advanced"],
                "default": "basic",
                "help": "Defect pipeline variant.",
            },
        ),
        ArgSpec(
            flags=("--data-parallel",),
            kwargs={
                "action": "store_true",
                "default": False,
                "help": "Wrap model in DataParallel when multiple GPUs are available.",
            },
        ),
        ArgSpec(
            flags=("--num-gpus",),
            kwargs={
                "type": int,
                "default": None,
                "help": "Number of GPUs to use from index 0 (sets CUDA_VISIBLE_DEVICES).",
            },
        ),
        ArgSpec(
            flags=("--gpu-ids",),
            kwargs={
                "type": str,
                "default": None,
                "help": "Comma-separated physical GPU ids (sets CUDA_VISIBLE_DEVICES).",
            },
        ),
        ArgSpec(
            flags=("--seed",),
            kwargs={"type": int, "default": 42},
        ),
        ArgSpec(
            flags=("--cache-dir",),
            kwargs={
                "type": str,
                "default": None,
                "help": "Directory for defect cache NPZ files. Auto-resolved if omitted.",
            },
        ),
        ArgSpec(
            flags=("--cache-read",),
            kwargs={
                "dest": "cache_read",
                "action": "store_true",
                "default": True,
                "help": "Read defect samples from cache if available (default: on).",
            },
        ),
        ArgSpec(
            flags=("--no-cache-read",),
            kwargs={
                "dest": "cache_read",
                "action": "store_false",
                "help": "Disable reading from defect cache.",
            },
        ),
        ArgSpec(
            flags=("--cache-write",),
            kwargs={
                "dest": "cache_write",
                "action": "store_true",
                "default": True,
                "help": "Write generated defect samples to cache (default: on).",
            },
        ),
        ArgSpec(
            flags=("--no-cache-write",),
            kwargs={
                "dest": "cache_write",
                "action": "store_false",
                "help": "Disable writing to defect cache.",
            },
        ),
        ArgSpec(
            flags=("--discord",),
            kwargs={
                "action": "store_true",
                "default": False,
                "help": "Enable Discord training notifications.",
            },
        ),
        ArgSpec(
            flags=("--discord-webhook-url",),
            kwargs={
                "type": str,
                "default": None,
                "help": "Discord webhook URL. If omitted, DISCORD_WEBHOOK_URL env var is used.",
            },
        ),
        ArgSpec(
            flags=("--discord-project-name",),
            kwargs={
                "type": str,
                "default": "BIT Thesis Project",
                "help": "Project name displayed in Discord notifications.",
            },
        ),
        ArgSpec(
            flags=("--discord-avatar-name",),
            kwargs={
                "type": str,
                "default": "Training Bot",
                "help": "Avatar/author name shown in Discord notifications.",
            },
        ),
        ArgSpec(
            flags=("--discord-progress-every",),
            kwargs={
                "type": int,
                "default": 1,
                "help": "Send Discord progress update every N epochs.",
            },
        ),
    ]


def main() -> None:
    preselected_gpu_ids = _preconfigure_cuda_visible_devices(sys.argv[1:])

    args, cfg = parse_and_bootstrap(
        schema=_build_schema(),
        data_subdir=None,
        description="Unified trainer for reconstruction models.",
    )

    if args.num_gpus is not None and args.num_gpus <= 0:
        raise ValueError("--num-gpus must be > 0")

    selected_gpu_ids = preselected_gpu_ids
    if args.gpu_ids:
        selected_gpu_ids = _parse_gpu_ids(args.gpu_ids)
    elif args.num_gpus is not None:
        selected_gpu_ids = list(range(int(args.num_gpus)))

    visible_gpus = torch.cuda.device_count() if cfg.device.type == "cuda" else 0
    auto_data_parallel = visible_gpus > 1
    enable_data_parallel = bool(args.data_parallel) or auto_data_parallel

    model_name = args.model.strip().lower()
    model_params = _default_model_params(model_name)
    model_params.update(_json_dict(args.model_params))

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
    run_name = args.run_name or f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoints_dir / args.output_model
    periodic_checkpoint_template = checkpoints_dir / "checkpoint_epoch_{epoch:04d}.pt"
    loss_curve_path = run_dir / "loss_curve.png"
    summary_path = run_dir / "training_summary.json"

    logger.info(f"Training model: {model_name}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Output run dir: {run_dir}")
    if selected_gpu_ids is not None:
        logger.info(f"Requested GPU ids: {selected_gpu_ids}")
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        logger.info(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"Visible CUDA GPUs: {visible_gpus}")
    if auto_data_parallel and not bool(args.data_parallel):
        logger.info("DataParallel enabled automatically (multiple GPUs selected)")

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

    dataset_factory = (
        create_advanced_reconstruction_dataset
        if args.dataset_variant == "advanced"
        else create_basic_reconstruction_dataset
    )

    cache_dir: str | None = None
    if bool(args.cache_read) or bool(args.cache_write):
        if args.cache_dir:
            cache_dir = str(Path(args.cache_dir).expanduser().resolve())
        else:
            cache_dir = str(cfg.data_dir / f"ShapeNetV2_{args.dataset_variant}_defected")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Defect cache: dir={cache_dir} "
            f"read={bool(args.cache_read)} write={bool(args.cache_write)}"
        )

    dataset = dataset_factory(
        root=str(cfg.data_dir / "ShapeNetV2"),
        seed=cfg.seed,
        defect_cache_npz_dir=cache_dir,
        defect_cache_read=bool(args.cache_read),
        defect_cache_write=bool(args.cache_write),
    )

    effective_batch_size = int(args.batch_size)
    effective_batch_size = _cap_batch_size_for_model(
        model_name, effective_batch_size, bool(args.overfit)
    )
    if args.overfit:
        overfit_count = min(max(1, int(args.overfit_samples)), len(dataset))
        dataset = Subset(dataset, list(range(overfit_count)))
        effective_batch_size = max(1, effective_batch_size // 2)
        effective_batch_size = _cap_batch_size_for_model(
            model_name, effective_batch_size, True
        )
        logger.warning(
            f"Overfit mode enabled: using {overfit_count} samples, "
            f"batch_size={effective_batch_size}"
        )

    train_loader, val_loader, _ = create_train_val_test_dataloaders(
        dataset=dataset,
        batch_size=effective_batch_size,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=cfg.seed,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    model = create_model(
        ModelConfig(name=model_name, params=model_params),
        device=cfg.device,
        data_parallel=enable_data_parallel,
        num_gpus=visible_gpus,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=int(args.lr_step_size),
        gamma=float(args.lr_gamma),
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_checkpoint:
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
        metrics = (
            loaded.get("metrics") if isinstance(loaded.get("metrics"), dict) else {}
        )
        best_val_loss = float(
            metrics.get("val_loss", loaded.get("val_loss", best_val_loss))
        )
        logger.info(
            f"Resumed from checkpoint: {args.resume_checkpoint} "
            f"(start_epoch={start_epoch}, best_val={best_val_loss:.6f})"
        )

    train_losses: list[float] = []
    val_losses: list[float] = []

    logger.info(
        f"Train size={len(train_loader.dataset)}, Val size={len(val_loader.dataset)}, "
        f"epochs={args.epochs}, lr={learning_rate}, wd={weight_decay}"
    )

    if notifier is not None:
        try:
            notifier.send_training_start(
                total_epochs=int(args.epochs),
                batch_size=effective_batch_size,
                train_size=len(train_loader.dataset),
                val_size=len(val_loader.dataset),
                training_on=str(cfg.device),
                number_of_gpus=visible_gpus,
                learning_rate=learning_rate,
            )
        except Exception as exc:
            logger.warning(f"Discord start notification failed: {exc}")

    wall_start = time.time()
    progress = tqdm(
        range(start_epoch, int(args.epochs) + 1), desc="Training", unit="epoch"
    )
    try:
        for epoch in progress:
            train_loss, train_metrics = _run_epoch(
                model_name=model_name,
                model=model,
                loader=train_loader,
                device=cfg.device,
                training=True,
                optimizer=optimizer,
                grad_clip=float(args.grad_clip),
                epoch=epoch,
                total_epochs=int(args.epochs),
            )
            with torch.no_grad():
                val_loss, val_metrics = _run_epoch(
                    model_name=model_name,
                    model=model,
                    loader=val_loader,
                    device=cfg.device,
                    training=False,
                )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step()

            _save_loss_plot(train_losses, val_losses, loss_curve_path)

            elapsed = time.time() - wall_start
            epochs_done = epoch - start_epoch + 1
            epochs_left = int(args.epochs) - epoch
            avg_epoch_time = elapsed / max(epochs_done, 1)
            eta_seconds = int(avg_epoch_time * max(epochs_left, 0))

            progress.set_postfix(
                train=f"{train_loss:.6f}",
                val=f"{val_loss:.6f}",
                best=f"{best_val_loss:.6f}",
                eta=f"{eta_seconds // 60}m {eta_seconds % 60}s",
            )

            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_coarse": train_metrics.get("coarse", 0.0),
                "train_fine": train_metrics.get("fine", 0.0),
                "val_coarse": val_metrics.get("coarse", 0.0),
                "val_fine": val_metrics.get("fine", 0.0),
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
                    metrics=epoch_metrics,
                    model_config=ModelConfig(name=model_name, params=model_params),
                    extra={"args": vars(args)},
                )
                logger.info(
                    f"Saved best checkpoint at epoch {epoch} with val_loss={val_loss:.6f}"
                )

            if (
                bool(args.save_checkpoints)
                and int(args.save_every) > 0
                and epoch % int(args.save_every) == 0
            ):
                periodic_path = Path(
                    str(periodic_checkpoint_template).format(epoch=epoch)
                )
                save_model_checkpoint(
                    checkpoint_path=periodic_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    model_config=ModelConfig(name=model_name, params=model_params),
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
                        batch_size=effective_batch_size,
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

    total_seconds = int(time.time() - wall_start)
    summary = {
        "model": model_name,
        "model_params": model_params,
        "epochs": int(args.epochs),
        "batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_val_loss": best_val_loss,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "loss_curve": str(loss_curve_path),
        "duration_seconds": total_seconds,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "visible_gpus": visible_gpus,
        "selected_gpu_ids": selected_gpu_ids,
        "data_parallel": enable_data_parallel,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Training finished in {total_seconds}s")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best checkpoint: {best_checkpoint_path}")
    logger.info(f"Run summary: {summary_path}")

    if notifier is not None:
        try:
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
