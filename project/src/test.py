from __future__ import annotations

import argparse

from core import (
    bootstrap,
    create_basic_reconstruction_dataset,
    load_model_checkpoint,
    create_model,
    ModelConfig,
)
import polyscope as ps
import torch
from pytorch3d.ops import knn_points
from metrics import (
    chamfer_distance_metric,
    density_aware_chamfer_distance_metric,
    hausdorff_distance_metric,
)


def _region_masks(a: torch.Tensor, b: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mask points in a whose nearest point in b is farther than threshold."""
    d2 = knn_points(a.unsqueeze(0), b.unsqueeze(0), K=1).dists.squeeze(0).squeeze(-1)
    return torch.sqrt(d2) > threshold


def split_original_defected(
    original: torch.Tensor,
    defected: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Split original/defected into repaired-target and preserved/noise regions.

    repaired_target corresponds to points present in original but missing in defected.
    """
    missing_mask = _region_masks(original, defected, threshold)
    defected_extra_mask = _region_masks(defected, original, threshold)

    return {
        "repaired_target": original[missing_mask],
        "original_preserved": original[~missing_mask],
        "defected_extra": defected[defected_extra_mask],
        "defected_supported": defected[~defected_extra_mask],
    }


def split_prediction_by_target_region(
    prediction: torch.Tensor,
    original: torch.Tensor,
    repaired_target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split predicted points by nearest GT region (repaired target vs preserved)."""
    nn_idx = (
        knn_points(prediction.unsqueeze(0), original.unsqueeze(0), K=1)
        .idx.squeeze(0)
        .squeeze(-1)
        .long()
    )
    pred_hits_repaired = repaired_target_mask[nn_idx]

    return {
        "pred_repaired": prediction[pred_hits_repaired],
        "pred_preserved": prediction[~pred_hits_repaired],
    }


def _run_model(
    model: torch.nn.Module, x: torch.Tensor, device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        _, pred = model(x.to(device))
        return pred.squeeze(0).cpu()


def _show_clouds(clouds: dict[str, torch.Tensor]) -> None:
    ps.init()
    ps.set_ground_plane_mode("none")

    color_map = {
        "original": (0.0, 1.0, 0.0),
        "defected": (1.0, 0.0, 0.0),
        "prediction": (0.0, 0.0, 1.0),
        "repaired_target": (1.0, 1.0, 0.0),
        "original_preserved": (0.3, 1.0, 0.3),
        "defected_extra": (1.0, 0.5, 0.0),
        "defected_supported": (0.8, 0.3, 0.3),
        "pred_repaired": (0.0, 1.0, 1.0),
        "pred_preserved": (0.2, 0.2, 1.0),
    }

    for name, points in clouds.items():
        if points is None or points.numel() == 0:
            continue

        ps.register_point_cloud(
            name,
            points,
            radius=0.00035,
            color=color_map.get(name, (0.8, 0.8, 0.8)),
            point_render_mode="quad",
        )

    ps.show()


def _compute_pair_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    density_alpha: float,
) -> dict[str, float]:
    if source.numel() == 0 or target.numel() == 0:
        return {
            "chamfer": float("nan"),
            "dcd": float("nan"),
            "hausdorff": float("nan"),
        }

    return {
        "chamfer": float(chamfer_distance_metric(source, target).item()),
        "dcd": float(
            density_aware_chamfer_distance_metric(
                source,
                target,
                alpha=density_alpha,
            ).item()
        ),
        "hausdorff": float(hausdorff_distance_metric(source, target).item()),
    }


def _print_pairwise_metrics(
    clouds: dict[str, torch.Tensor],
    density_alpha: float,
) -> None:
    # Core comparisons requested in comment.
    pairs = [
        ("original", "defected", "damage severity"),
        ("original", "prediction", "overall reconstruction quality"),
        (
            "original_preserved",
            "pred_preserved",
            "preserved-region fidelity",
        ),
        (
            "repaired_target",
            "pred_repaired",
            "repair-region quality",
        ),
        # Extra diagnostics that are often useful.
        ("defected", "prediction", "total change from defected input"),
        (
            "defected_extra",
            "pred_repaired",
            "whether model over-focuses on artifact region",
        ),
    ]

    print("\nPairwise metrics (lower is better):")
    for src_name, tgt_name, note in pairs:
        values = _compute_pair_metrics(
            source=clouds[src_name],
            target=clouds[tgt_name],
            density_alpha=density_alpha,
        )

        print(
            f"- {src_name} vs {tgt_name} ({note}) | "
            f"CD={values['chamfer']:.6f}, DCD={values['dcd']:.6f}, HD={values['hausdorff']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Segmented repair-target extraction test"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--segment-threshold", type=float, default=0.02)
    parser.add_argument("--density-alpha", type=float, default=1000.0)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    cfg = bootstrap()

    model = create_model(
        ModelConfig(
            name="pointr",
            params={
                "trans_dim": 384,
                "knn_layer": 1,
                "num_pred": 16384,
                "num_query": 224,
            },
        ),
        device=cfg.device,
    )

    load_model_checkpoint(
        checkpoint_path=cfg.checkpoint_dir / "pointr" / "v1_best.pt",
        model=model,
        map_location=cfg.device,
        strict=True,
        weights_only=False,
    )
    model.eval()

    basic_dataset = create_basic_reconstruction_dataset(
        seed=cfg.seed, root=cfg.data_dir
    )

    if args.sample_index < 0 or args.sample_index >= len(basic_dataset):
        raise ValueError(
            f"--sample-index must be in [0, {len(basic_dataset) - 1}], got {args.sample_index}"
        )

    sample = basic_dataset[args.sample_index]
    original = sample.original_pos.float().cpu()
    defected = sample.defected_pos.float().cpu()

    prediction = _run_model(model, defected.unsqueeze(0), cfg.device)

    split_gt = split_original_defected(
        original=original,
        defected=defected,
        threshold=args.segment_threshold,
    )

    repaired_target_mask = _region_masks(original, defected, args.segment_threshold)
    split_pred = split_prediction_by_target_region(
        prediction=prediction,
        original=original,
        repaired_target_mask=repaired_target_mask,
    )

    print(f"Sample index: {args.sample_index}")
    print(f"Threshold: {args.segment_threshold}")
    print(f"Original points: {original.shape[0]}")
    print(f"Defected points: {defected.shape[0]}")
    print(f"Prediction points: {prediction.shape[0]}")
    print(
        f"Repaired target points (original - defected): {split_gt['repaired_target'].shape[0]}"
    )
    print(f"Original preserved points: {split_gt['original_preserved'].shape[0]}")
    print(
        f"Defected extra points (defected - original): {split_gt['defected_extra'].shape[0]}"
    )
    print(f"Defected supported points: {split_gt['defected_supported'].shape[0]}")
    print(
        f"Pred points mapped to repaired target: {split_pred['pred_repaired'].shape[0]}"
    )
    print(
        f"Pred points mapped to preserved region: {split_pred['pred_preserved'].shape[0]}"
    )

    clouds = {
        "original": original,  # whole original cloud
        "defected": defected,  # whole defected cloud
        "prediction": prediction,  # whole predicted cloud
        "repaired_target": split_gt[
            "repaired_target"
        ],  # points in original missing from defected
        "original_preserved": split_gt[
            "original_preserved"
        ],  # points in original preserved in defected
        "defected_extra": split_gt[
            "defected_extra"
        ],  # points in defected not in original
        "defected_supported": split_gt[
            "defected_supported"
        ],  # points in defected supported by original
        "pred_repaired": split_pred[
            "pred_repaired"
        ],  # predicted points closest to repaired target region
        "pred_preserved": split_pred[
            "pred_preserved"
        ],  # predicted points closest to preserved region
    }

    _print_pairwise_metrics(clouds, density_alpha=args.density_alpha)

    if args.visualize:
        _show_clouds(clouds)


if __name__ == "__main__":
    main()
