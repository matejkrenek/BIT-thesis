"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: model_defaults.py
Responsibility: Centralized default model configuration and training hyperparameters for CLI scripts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_MODEL_PARAMS_DEFAULTS: dict[str, dict[str, Any]] = {
    "pcn": {
        "num_dense": 16384,
        "latent_dim": 1024,
        "grid_size": 4,
    },
    "pointr": {
        "trans_dim": 384,
        "knn_layer": 1,
        "num_pred": 16384,
        "num_query": 224,
    },
    "adapointr": {
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
    },
    "pointcleannet": {
        "num_points": 500,
        "num_scales": 1,
        "output_dim": 3,
        "use_point_stn": True,
        "use_feat_stn": True,
        "sym_op": "max",
        "point_tuple": 1,
    },
    "pointcleannetoutliers": {
        "num_points": 500,
        "num_scales": 1,
        "output_dim": 1,
        "use_point_stn": True,
        "use_feat_stn": True,
        "sym_op": "max",
        "point_tuple": 1,
    },
}


_OPTIMIZER_DEFAULTS: dict[str, dict[str, float]] = {
    "pcn": {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    },
    "pointr": {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    },
    "adapointr": {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    },
}


def _normalize_model_name(model_name: str) -> str:
    return str(model_name).strip().lower().replace("-", "").replace("_", "")


def get_default_model_params(model_name: str) -> dict[str, Any]:
    """Return a deep copy of default constructor params for a supported model."""
    normalized = _normalize_model_name(model_name)
    if normalized not in _MODEL_PARAMS_DEFAULTS:
        raise ValueError(f"Unsupported model '{model_name}'")
    return deepcopy(_MODEL_PARAMS_DEFAULTS[normalized])


def get_default_optimizer_hparams(model_name: str) -> dict[str, float]:
    """Return a copy of default optimizer hyperparameters for a supported model."""
    normalized = _normalize_model_name(model_name)
    if normalized not in _OPTIMIZER_DEFAULTS:
        raise ValueError(f"Unsupported model '{model_name}'")
    return dict(_OPTIMIZER_DEFAULTS[normalized])


def get_default_learning_rate(model_name: str) -> float:
    """Return default learning rate for a supported model."""
    return float(get_default_optimizer_hparams(model_name)["learning_rate"])


def get_default_weight_decay(model_name: str) -> float:
    """Return default weight decay for a supported model."""
    return float(get_default_optimizer_hparams(model_name)["weight_decay"])


__all__ = [
    "get_default_model_params",
    "get_default_optimizer_hparams",
    "get_default_learning_rate",
    "get_default_weight_decay",
]
