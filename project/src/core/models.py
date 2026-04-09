from __future__ import annotations

import importlib
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    """Universal model creation config.

    Example:
        ModelConfig(name="pcn", params={"num_dense": 16384})
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


def _normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _build_pcn(params: Mapping[str, Any]) -> nn.Module:
    from models.pcn import PCN

    return PCN(**dict(params))


def _build_pointr(params: Mapping[str, Any]) -> nn.Module:
    from models.pointr import PoinTr

    config = SimpleNamespace(**dict(params))
    return PoinTr(config=config)


class _ConfigNode(dict):
    """Dict with attribute access, recursive for nested config trees."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_config_node(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return _ConfigNode({k: _to_config_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config_node(v) for v in value]
    return value


def _build_adapointr(params: Mapping[str, Any]) -> nn.Module:
    from models.adapointr.model import AdaPoinTr

    config = _to_config_node(dict(params))
    return AdaPoinTr(config=config)


def _build_pointcleannet_hybrid(params: Mapping[str, Any]) -> nn.Module:
    from models.pointcleannet import PointCleanNetHybrid

    return PointCleanNetHybrid(**dict(params))


_MODEL_BUILDERS = {
    "pcn": _build_pcn,
    "pointr": _build_pointr,
    "adapointr": _build_adapointr,
    "pointcleannethybrid": _build_pointcleannet_hybrid,
}


def available_models() -> list[str]:
    """Return supported model aliases for the factory."""
    return sorted(_MODEL_BUILDERS.keys())


def create_model(
    config: ModelConfig | str,
    params: Mapping[str, Any] | None = None,
    *,
    device: str | torch.device | None = None,
    data_parallel: bool = False,
    num_gpus: int | None = None,
) -> nn.Module:
    """Create a model instance from a universal config.

    - ``config`` can be ModelConfig or model-name string.
    - ``params`` are used only when ``config`` is a model-name string.
    """
    if isinstance(config, ModelConfig):
        model_name = config.name
        model_params = dict(config.params)
    else:
        model_name = config
        model_params = dict(params or {})

    normalized = _normalize_model_name(model_name)
    if normalized not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(available_models())}"
        )

    model = _MODEL_BUILDERS[normalized](model_params)

    if data_parallel:
        gpu_count = num_gpus if num_gpus is not None else torch.cuda.device_count()
        if gpu_count > 1:
            model = nn.DataParallel(model)

    if device is not None:
        model = model.to(device)

    return model


def _model_state_dict(model: nn.Module) -> dict[str, Any]:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _load_model_state_dict(
    model: nn.Module, state: Mapping[str, Any], *, strict: bool
) -> None:
    if hasattr(model, "module"):
        model.module.load_state_dict(state, strict=strict)
    else:
        model.load_state_dict(state, strict=strict)


def save_model_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    step: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    model_key: str = "model_state",
) -> Path:
    """Save model (and optional training state) into a single checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        model_key: _model_state_dict(model),
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    if step is not None:
        checkpoint["step"] = int(step)
    if metrics is not None:
        checkpoint["metrics"] = dict(metrics)
    if model_config is not None:
        if isinstance(model_config, ModelConfig):
            checkpoint["model_config"] = {
                "name": model_config.name,
                "params": dict(model_config.params),
            }
        else:
            checkpoint["model_config"] = dict(model_config)
    if extra is not None:
        checkpoint["extra"] = dict(extra)

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_model_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
    model_key: str = "model_state",
    weights_only: bool = True,
) -> dict[str, Any]:
    """Load model state and optional optimizer/scheduler states from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except TypeError:
        # Backward compatibility for PyTorch builds without weights_only support.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, MappingABC):
        raise TypeError(
            f"Checkpoint '{checkpoint_path}' must be a mapping, got {type(checkpoint).__name__}."
        )

    state = checkpoint.get(model_key)
    if state is None:
        # Compatibility fallback for checkpoints that are plain state_dict.
        if all(isinstance(k, str) for k in checkpoint.keys()):
            state = checkpoint
        else:
            raise KeyError(
                f"Checkpoint '{checkpoint_path}' does not contain key '{model_key}'."
            )

    _load_model_state_dict(model, state, strict=strict)

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def create_and_load_model(
    model_config: ModelConfig,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    data_parallel: bool = False,
    num_gpus: int | None = None,
    strict: bool = True,
    map_location: str | torch.device | None = None,
    weights_only: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Create a model from config and immediately load checkpoint state."""
    model = create_model(
        model_config,
        device=device,
        data_parallel=data_parallel,
        num_gpus=num_gpus,
    )
    checkpoint = load_model_checkpoint(
        checkpoint_path,
        model,
        map_location=map_location if map_location is not None else device,
        strict=strict,
        weights_only=weights_only,
    )
    return model, checkpoint


__all__ = [
    "ModelConfig",
    "available_models",
    "create_model",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "create_and_load_model",
]
