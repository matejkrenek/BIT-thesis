from __future__ import annotations

import torch
import torch.nn as nn

from .pcpnet import ResMSPCPNet, ResPCPNet


class PointCleanNet(nn.Module):
    """PointCleanNet wrapper adapted for this repository's model API.

    Expected input shape: (B, N, 3). Internally converted to (B, 3, N).
    Output is a tuple: (pred_clean_point, patch_rotation), where:
      - pred_clean_point has shape (B, 3)
      - patch_rotation has shape (B, 3, 3) or None
    """

    def __init__(
        self,
        num_points: int = 500,
        num_scales: int = 1,
        output_dim: int = 3,
        use_point_stn: bool = True,
        use_feat_stn: bool = True,
        sym_op: str = "max",
        get_pointfvals: bool = False,
        point_tuple: int = 1,
    ):
        super().__init__()
        self.num_points = int(num_points)

        if int(num_scales) <= 1:
            self.backbone = ResPCPNet(
                num_points=self.num_points,
                output_dim=int(output_dim),
                use_point_stn=bool(use_point_stn),
                use_feat_stn=bool(use_feat_stn),
                sym_op=str(sym_op),
                get_pointfvals=bool(get_pointfvals),
                point_tuple=int(point_tuple),
            )
        else:
            self.backbone = ResMSPCPNet(
                num_scales=int(num_scales),
                num_points=self.num_points,
                output_dim=int(output_dim),
                use_point_stn=bool(use_point_stn),
                use_feat_stn=bool(use_feat_stn),
                sym_op=str(sym_op),
                get_pointfvals=bool(get_pointfvals),
                point_tuple=int(point_tuple),
            )

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"Expected input shape (B, N, 3), got {tuple(xyz.shape)}")
        if int(xyz.shape[1]) != self.num_points:
            raise ValueError(
                f"PointCleanNet expects N={self.num_points} points per patch, "
                f"got N={int(xyz.shape[1])}."
            )

        x = xyz.transpose(2, 1).contiguous()  # (B, 3, N)
        pred, patch_rot, _, _ = self.backbone(x)
        return pred, patch_rot

    @staticmethod
    def _surface_distance(
        prediction: torch.Tensor, target_patch: torch.Tensor
    ) -> torch.Tensor:
        """PointCleanNet-style point-to-surface distance with max-distance stabilizer."""
        if prediction.ndim != 2 or prediction.shape[1] != 3:
            raise ValueError(
                f"prediction must have shape (B, 3), got {tuple(prediction.shape)}"
            )
        if target_patch.ndim != 3 or target_patch.shape[2] != 3:
            raise ValueError(
                f"target_patch must have shape (B, K, 3), got {tuple(target_patch.shape)}"
            )

        m2 = prediction.unsqueeze(1).expand(-1, target_patch.shape[1], -1)
        m = (target_patch - m2).pow(2).sum(dim=2)

        min_dist = torch.min(m, dim=1)[0]
        max_dist = torch.max(m, dim=1)[0]

        alpha = 0.99
        dist = torch.mean(alpha * min_dist + (1.0 - alpha) * max_dist)
        return dist * 100.0

    def get_loss(self, pred, target_patch: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, (tuple, list)):
            point_pred = pred[0]
            patch_rot = pred[1] if len(pred) > 1 else None
        else:
            point_pred = pred
            patch_rot = None

        if patch_rot is not None:
            # QSTN inverse for rotations is transpose.
            point_pred = torch.bmm(
                point_pred.unsqueeze(1), patch_rot.transpose(2, 1)
            ).squeeze(1)

        return self._surface_distance(point_pred, target_patch)

    def compute_loss(self, pred, target_patch: torch.Tensor) -> torch.Tensor:
        return self.get_loss(pred, target_patch)


class PointCleanNetOutliers(PointCleanNet):
    """PointCleanNet outlier-removal head.

    The network predicts one logit per patch (center-point outlier probability).
    """

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_dim"] = 1
        super().__init__(*args, **kwargs)
        self._bce = nn.BCEWithLogitsLoss()

    def get_outlier_loss(self, pred, target_label: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, (tuple, list)):
            logits = pred[0]
        else:
            logits = pred

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim != 1:
            raise ValueError(
                f"Outlier logits must have shape (B,) or (B,1), got {tuple(logits.shape)}"
            )

        if target_label.ndim == 2 and target_label.shape[1] == 1:
            target_label = target_label[:, 0]
        elif target_label.ndim != 1:
            raise ValueError(
                "Outlier labels must have shape (B,) or (B,1), "
                f"got {tuple(target_label.shape)}"
            )

        target_label = target_label.float()
        return self._bce(logits, target_label)

    def get_loss(self, pred, target_label: torch.Tensor) -> torch.Tensor:
        return self.get_outlier_loss(pred, target_label)

    def compute_loss(self, pred, target_label: torch.Tensor) -> torch.Tensor:
        return self.get_outlier_loss(pred, target_label)
