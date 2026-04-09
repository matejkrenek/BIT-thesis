from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn

from .legacy.noise_removal import pcpnet as local_noise_pcpnet
from .legacy.outliers_removal import pcpnet as local_outliers_pcpnet


@dataclass
class PointCleanNetStageConfig:
    stage_type: Literal["outliers", "noise"]
    checkpoint_file: Optional[Path]
    points_per_patch: int = 500
    output_dim: int = 3
    use_point_stn: bool = True
    use_feat_stn: bool = True
    sym_op: str = "max"
    point_tuple: int = 1


class _PointCleanNetStage(nn.Module):
    def __init__(self, cfg: PointCleanNetStageConfig):
        super().__init__()
        if cfg.stage_type == "outliers":
            pcpnet_module = local_outliers_pcpnet
        elif cfg.stage_type == "noise":
            pcpnet_module = local_noise_pcpnet
        else:
            raise ValueError(f"Unsupported stage_type: {cfg.stage_type}")

        model_cls = getattr(pcpnet_module, "ResPCPNet", None)
        if model_cls is None:
            raise AttributeError(
                "ResPCPNet was not found in vendored PointCleanNet module"
            )

        self.model = model_cls(
            num_points=cfg.points_per_patch,
            output_dim=cfg.output_dim,
            use_point_stn=cfg.use_point_stn,
            use_feat_stn=cfg.use_feat_stn,
            sym_op=cfg.sym_op,
            point_tuple=cfg.point_tuple,
        )

        if cfg.checkpoint_file is not None:
            state = torch.load(
                cfg.checkpoint_file, map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(state, strict=True)
        self.use_point_stn = bool(cfg.use_point_stn)

    def forward(
        self, patch_points_bnk3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Expect (B, N, 3), convert to PointNet layout (B, 3, N).
        x = patch_points_bnk3.transpose(1, 2)
        pred, trans, _, _ = self.model(x)
        return pred, trans


class PointCleanNetHybrid(nn.Module):
    """Combined PointCleanNet model with outlier and denoise branches.

    - No dependency on libs/pointcleannet; uses vendored local implementation.
    - Checkpoints are optional. If provided, they initialize the corresponding branch.
    - Expects precomputed normalized patches as input.
    - The forward pass returns outputs for both tasks so the model can be trained end-to-end.
    """

    def __init__(
        self,
        outlier_checkpoint_file: str | None = None,
        denoise_checkpoint_file: str | None = None,
        points_per_patch: int = 500,
        outlier_threshold: float = 0.5,
        min_kept_points: int = 32,
    ):
        super().__init__()

        self.points_per_patch = int(points_per_patch)
        self.outlier_threshold = float(outlier_threshold)
        self.min_kept_points = int(min_kept_points)

        if self.points_per_patch <= 0:
            raise ValueError("points_per_patch must be > 0")

        self.outlier_stage = _PointCleanNetStage(
            PointCleanNetStageConfig(
                stage_type="outliers",
                checkpoint_file=(
                    Path(outlier_checkpoint_file) if outlier_checkpoint_file else None
                ),
                points_per_patch=self.points_per_patch,
                output_dim=1,
            )
        )

        self.denoise_stage = _PointCleanNetStage(
            PointCleanNetStageConfig(
                stage_type="noise",
                checkpoint_file=(
                    Path(denoise_checkpoint_file) if denoise_checkpoint_file else None
                ),
                points_per_patch=self.points_per_patch,
                output_dim=3,
            )
        )

    @staticmethod
    def _apply_inverse_stn(
        pred: torch.Tensor, trans: torch.Tensor | None
    ) -> torch.Tensor:
        if trans is None:
            return pred
        return torch.bmm(pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)

    def _forward_single(self, patches: torch.Tensor) -> dict[str, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[2] != 3:
            raise ValueError(
                "Expected normalized patches with shape (P, K, 3), "
                f"got {tuple(patches.shape)}"
            )
        if patches.shape[1] != self.points_per_patch:
            raise ValueError(
                f"Expected K={self.points_per_patch} points per patch, "
                f"got K={int(patches.shape[1])}"
            )

        outlier_pred, _ = self.outlier_stage(patches)
        outlier_scores = outlier_pred.squeeze(-1)
        inlier_mask = outlier_scores <= self.outlier_threshold

        if int(inlier_mask.sum().item()) < self.min_kept_points:
            keep = min(max(self.min_kept_points, 1), int(patches.shape[0]))
            keep_idx = torch.argsort(outlier_scores, descending=False)[:keep]
            inlier_mask = torch.zeros_like(outlier_scores, dtype=torch.bool)
            inlier_mask[keep_idx] = True

        # Denoise all incoming patches so both branches can be trained jointly.
        denoise_pred, denoise_trans = self.denoise_stage(patches)
        denoise_pred = self._apply_inverse_stn(denoise_pred, denoise_trans)

        kept_patch_indices = torch.nonzero(inlier_mask, as_tuple=False).squeeze(1)

        return {
            "input_patches": patches,
            "outlier_scores": outlier_scores,
            "inlier_mask": inlier_mask,
            "kept_patch_indices": kept_patch_indices,
            "denoise_displacements": denoise_pred,
            "denoise_displacements_inlier": denoise_pred[inlier_mask],
        }

    def forward(
        self, patches: torch.Tensor
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        # Accept both (P,K,3) and (B,P,K,3).
        if patches.ndim == 3:
            return self._forward_single(patches)
        if patches.ndim == 4:
            return [self._forward_single(patches[i]) for i in range(patches.shape[0])]
        raise ValueError(
            f"Expected input shape (P,K,3) or (B,P,K,3), got {tuple(patches.shape)}"
        )
