from __future__ import annotations

import argparse
import numpy as np

from core import (
    bootstrap,
    create_basic_reconstruction_dataset,
)
import polyscope as ps
import torch


def _show_clouds(clouds: dict[str, torch.Tensor]) -> None:
    ps.init()
    ps.set_ground_plane_mode("none")

    color_map = {
        "original": (0.0, 1.0, 0.0),
        "defected": (1.0, 0.0, 0.0),
        "all_patch_original": (0.1, 0.7, 0.7),
        "all_patch_defected": (0.8, 0.1, 0.8),
        "all_patch_centers": (1.0, 1.0, 0.0),
        "patch_original": (0.0, 0.8, 0.8),
        "patch_defected": (1.0, 0.0, 1.0),
        "patch_center": (1.0, 1.0, 1.0),
    }

    for name, points in clouds.items():
        if points is None or points.numel() == 0:
            continue

        ps_points = _to_polyscope_points(points)

        ps.register_point_cloud(
            name,
            ps_points,
            radius=0.00035,
            color=color_map.get(name, (0.8, 0.8, 0.8)),
            point_render_mode="quad",
        )

    ps.show()


def _extract_patch(
    sample,
    key: str,
    patch_index: int,
    valid_count_key: str | None = None,
) -> torch.Tensor:
    patches = getattr(sample, key)
    if not torch.is_tensor(patches) or patches.ndim != 3:
        raise ValueError(f"Expected {key} to have shape (M, K, 3)")

    m = int(patches.shape[0])
    if m == 0:
        raise ValueError(f"No patches available in {key}")

    patch_index = max(0, min(int(patch_index), m - 1))
    patch = patches[patch_index].float().cpu()

    if valid_count_key is None or not hasattr(sample, valid_count_key):
        return patch

    valid_counts = getattr(sample, valid_count_key)
    if not torch.is_tensor(valid_counts) or valid_counts.ndim != 1:
        return patch

    valid = int(valid_counts[patch_index].item())
    valid = max(0, min(valid, int(patch.shape[0])))
    return patch[:valid]


def _to_polyscope_points(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(points):
        arr = points.detach().cpu().numpy()
    else:
        arr = np.asarray(points)

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.reshape(-1, 3)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Polyscope expects shape (N, 3), got {arr.shape}")

    return arr


def _flatten_patches(
    patches: torch.Tensor,
    valid_counts: torch.Tensor | None,
) -> torch.Tensor:
    if not torch.is_tensor(patches) or patches.ndim != 3:
        raise ValueError("Expected patches with shape (M, K, 3)")

    out = []
    for i in range(int(patches.shape[0])):
        patch = patches[i].float().cpu()
        if valid_counts is not None and torch.is_tensor(valid_counts):
            valid = int(valid_counts[i].item())
            valid = max(0, min(valid, int(patch.shape[0])))
            patch = patch[:valid]
        out.append(patch)

    if not out:
        return torch.empty((0, 3), dtype=patches.dtype)
    return torch.cat(out, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Patch extraction test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--patching-method",
        type=str,
        default="pointcleannet_radius",
        choices=["fps_knn", "pointcleannet_radius"],
    )
    parser.add_argument("--patch-size", type=int, default=8192)
    parser.add_argument("--num-patches", type=int, default=64)
    parser.add_argument("--patch-radius", type=float, default=0.1)
    parser.add_argument(
        "--patch-center",
        type=str,
        default="point",
        choices=["point", "mean", "none"],
    )
    parser.add_argument("--patch-point-count-std", type=float, default=0.0)
    parser.add_argument("--patch-index", type=int, default=0)
    parser.add_argument("--normalize-patches", action="store_true")
    args = parser.parse_args()

    cfg = bootstrap()

    patched_dataset = create_basic_reconstruction_dataset(
        seed=cfg.seed,
        root=cfg.data_dir,
        dense=True,
        dense_root=cfg.data_dir / "ShapeNetV2_dense",
        split_into_patches=True,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        normalize_patches=args.normalize_patches,
        patching_method=args.patching_method,
        patch_radius=args.patch_radius,
        patch_center=args.patch_center,
        patch_point_count_std=args.patch_point_count_std,
        include_full_objects_in_patches=True,
    )

    if args.sample_index < 0 or args.sample_index >= len(patched_dataset):
        raise ValueError(
            f"--sample-index must be in [0, {len(patched_dataset) - 1}], got {args.sample_index}"
        )

    from core import create_model, ModelConfig, load_model_checkpoint

    pointr = create_model(
        ModelConfig(
            name="pointr",
            params={
                "trans_dim": 384,
                "knn_layer": 1,
                "num_pred": 16384,
                "num_query": 224,
            },
        )
    )

    load_model_checkpoint(
        checkpoint_path=cfg.checkpoint_dir / "pointr" / "v1_best.pt",
        model=pointr,
        map_location="cuda",
        strict=True,
        weights_only=False,
    )

    pointr.eval()

    _, pred = pointr(
        patched_dataset[args.sample_index].defected_pos.float().cpu()[0].unsqueeze(0)
    )

    ps.init()

    ps.register_point_cloud(
        "pred",
        _to_polyscope_points(pred.squeeze(0)),
        radius=0.00035,
        color=(0.0, 0.0, 1.0),
        point_render_mode="quad",
    )

    ps.register_point_cloud(
        "defected",
        patched_dataset[args.sample_index].defected_pos[0],
        radius=0.00035,
        color=(1.0, 0.0, 0.0),
        point_render_mode="quad",
    )

    ps.register_point_cloud(
        "original",
        patched_dataset[args.sample_index].original_pos[0],
        radius=0.00035,
        color=(0.0, 1.0, 0.0),
        point_render_mode="quad",
    )

    ps.show()

    print(pred)

    print(patched_dataset[0])
    exit(0)

    patched_sample = patched_dataset[args.sample_index]
    if hasattr(patched_sample, "original_full_pos") and hasattr(
        patched_sample, "defected_full_pos"
    ):
        original = patched_sample.original_full_pos.float().cpu()
        defected = patched_sample.defected_full_pos.float().cpu()
    else:
        # Fallback for configurations without full-object attachment.
        original = patched_sample.original_pos.float().cpu().reshape(-1, 3)
        defected = patched_sample.defected_pos.float().cpu().reshape(-1, 3)

    print(f"Sample index: {args.sample_index}")
    print(f"Original points: {original.shape[0]}")
    print(f"Defected points: {defected.shape[0]}")

    clouds = {
        "original": original,  # whole original cloud
        "defected": defected,  # whole defected cloud
    }

    all_patch_original = _flatten_patches(
        patched_sample.original_pos,
        getattr(patched_sample, "original_valid_counts", None),
    )
    all_patch_defected = _flatten_patches(
        patched_sample.defected_pos,
        getattr(patched_sample, "defected_valid_counts", None),
    )
    patch_original = _extract_patch(
        patched_sample,
        key="original_pos",
        patch_index=args.patch_index,
        valid_count_key="original_valid_counts",
    )
    patch_defected = _extract_patch(
        patched_sample,
        key="defected_pos",
        patch_index=args.patch_index,
        valid_count_key="defected_valid_counts",
    )

    safe_patch_index = max(
        0,
        min(int(args.patch_index), int(patched_sample.original_pos.shape[0]) - 1),
    )
    center = patched_sample.patch_centers[safe_patch_index].float().cpu().unsqueeze(0)

    print("\nPatch extraction summary:")
    print(f"- method: {patched_sample.patching_method}")
    print(f"- num_patches: {patched_sample.num_patches}")
    print(f"- patch_size: {patched_sample.patch_size}")
    print(f"- selected_patch_index: {safe_patch_index}")
    print(f"- patch_radius_ratio: {patched_sample.patch_radius}")
    print(f"- coverage_ratio: {patched_sample.coverage_ratio:.4f}")
    if hasattr(patched_sample, "original_full_pos") and hasattr(
        patched_sample, "defected_full_pos"
    ):
        print(f"- full_original_points: {patched_sample.original_full_pos.shape[0]}")
        print(f"- full_defected_points: {patched_sample.defected_full_pos.shape[0]}")
    print(f"- all_original_patch_points: {all_patch_original.shape[0]}")
    print(f"- all_defected_patch_points: {all_patch_defected.shape[0]}")
    print(f"- selected_original_patch_points: {patch_original.shape[0]}")
    print(f"- selected_defected_patch_points: {patch_defected.shape[0]}")

    clouds["all_patch_original"] = all_patch_original
    clouds["all_patch_defected"] = all_patch_defected
    clouds["all_patch_centers"] = patched_sample.patch_centers.float().cpu()
    clouds["patch_original"] = patch_original
    clouds["patch_defected"] = patch_defected
    clouds["patch_center"] = center

    if args.visualize:
        _show_clouds(clouds)


if __name__ == "__main__":
    main()
