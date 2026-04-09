from __future__ import annotations

import argparse
import numpy as np
import polyscope as ps
import torch

from core import (
    bootstrap,
    create_advanced_reconstruction_dataset,
)


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


def _bbox_diag(points: torch.Tensor) -> float:
    return float(
        torch.norm(points.max(dim=0).values - points.min(dim=0).values, p=2).item()
    )


def _iter_valid_patches(
    patches: torch.Tensor,
    valid_counts: torch.Tensor | None,
):
    if not torch.is_tensor(patches) or patches.ndim != 3:
        raise ValueError("Expected patches with shape (M, K, 3)")

    for i in range(int(patches.shape[0])):
        patch = patches[i].float().cpu()
        if valid_counts is not None and torch.is_tensor(valid_counts):
            valid = int(valid_counts[i].item())
            valid = max(0, min(valid, int(patch.shape[0])))
            patch = patch[:valid]
        yield i, patch


def _assemble_world_points(
    *,
    sample,
    normalized_patches: torch.Tensor,
    valid_counts: torch.Tensor | None,
    patch_center_mode: str,
) -> torch.Tensor:
    if not hasattr(sample, "patch_centers"):
        raise ValueError("Sample does not contain patch_centers metadata")
    if not hasattr(sample, "defected_full_pos"):
        raise ValueError(
            "Sample does not contain defected_full_pos; run dataset with include_full_objects_in_patches=True"
        )

    centers = torch.as_tensor(sample.patch_centers).float().cpu()
    full = torch.as_tensor(sample.defected_full_pos).float().cpu()
    radius_ratio = float(getattr(sample, "patch_radius", 0.0))

    if radius_ratio <= 0.0:
        raise ValueError("patch_radius metadata must be > 0 for denormalization")

    radius_abs = max(_bbox_diag(full) * radius_ratio, 1e-8)
    assembled = []

    for i, patch in _iter_valid_patches(normalized_patches, valid_counts):
        if patch.numel() == 0:
            continue

        if patch_center_mode == "point":
            world_patch = patch * radius_abs + centers[i].unsqueeze(0)
        elif patch_center_mode == "none":
            world_patch = patch * radius_abs
        elif patch_center_mode == "mean":
            # Exact inverse for mean-centering is not available without saving per-patch mean.
            world_patch = patch * radius_abs + centers[i].unsqueeze(0)
        else:
            raise ValueError(f"Unknown patch_center mode: {patch_center_mode}")

        assembled.append(world_patch)

    if not assembled:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.cat(assembled, dim=0)


def _deduplicate_points(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    if points.numel() == 0:
        return points
    if voxel_size <= 0:
        return points

    arr = points.detach().cpu().numpy()
    q = np.round(arr / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(q, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    return torch.from_numpy(arr[keep_idx]).to(dtype=points.dtype)


def _show_clouds(clouds: dict[str, torch.Tensor], point_radius: float) -> None:
    ps.init()
    ps.set_ground_plane_mode("none")

    color_map = {
        "original_full": (0.0, 1.0, 0.0),
        "defected_full": (1.0, 0.0, 0.0),
        "patch_centers": (1.0, 1.0, 0.0),
        "all_normalized_defected_patches": (0.2, 0.6, 1.0),
        "assembled_from_patches": (1.0, 0.4, 0.0),
        "assembled_deduplicated": (0.8, 0.9, 0.2),
    }

    for name, points in clouds.items():
        if points is None or points.numel() == 0:
            continue
        ps.register_point_cloud(
            name,
            _to_polyscope_points(points),
            radius=point_radius,
            color=color_map.get(name, (0.8, 0.8, 0.8)),
            point_render_mode="quad",
        )

    ps.show()


def main():
    parser = argparse.ArgumentParser(description="Patch assembly test (no inference)")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--patching-method",
        type=str,
        default="pointcleannet_radius",
        choices=["fps_knn", "pointcleannet_radius"],
    )
    parser.add_argument("--patch-size", type=int, default=4096)
    parser.add_argument("--num-patches", type=int, default=100)
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
    parser.add_argument("--dedup-voxel", type=float, default=0.0)
    parser.add_argument("--point-radius", type=float, default=0.00035)
    args = parser.parse_args()

    cfg = bootstrap()

    patched_dataset = create_advanced_reconstruction_dataset(
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

    patched_sample = patched_dataset[args.sample_index]
    if not hasattr(patched_sample, "original_full_pos") or not hasattr(
        patched_sample, "defected_full_pos"
    ):
        raise ValueError(
            "Patched sample does not have full-object tensors. "
            "Use include_full_objects_in_patches=True."
        )

    original_full = torch.as_tensor(patched_sample.original_full_pos).float().cpu()
    defected_full = torch.as_tensor(patched_sample.defected_full_pos).float().cpu()
    centers = torch.as_tensor(patched_sample.patch_centers).float().cpu()

    all_norm_defected = []
    for _, p in _iter_valid_patches(
        patched_sample.defected_pos,
        getattr(patched_sample, "defected_valid_counts", None),
    ):
        if p.numel() > 0:
            all_norm_defected.append(p)
    all_norm_defected = (
        torch.cat(all_norm_defected, dim=0)
        if all_norm_defected
        else torch.empty((0, 3), dtype=torch.float32)
    )

    assembled = _assemble_world_points(
        sample=patched_sample,
        normalized_patches=patched_sample.defected_pos,
        valid_counts=getattr(patched_sample, "defected_valid_counts", None),
        patch_center_mode=args.patch_center,
    )
    assembled_dedup = _deduplicate_points(assembled, voxel_size=float(args.dedup_voxel))

    print(f"Sample index: {args.sample_index}")
    print("\nPatch assembly summary:")
    print(f"- method: {patched_sample.patching_method}")
    print(f"- num_patches: {int(patched_sample.num_patches)}")
    print(f"- patch_size: {int(patched_sample.patch_size)}")
    print(f"- patch_center: {args.patch_center}")
    print(f"- patch_radius_ratio: {float(patched_sample.patch_radius):.6f}")
    print(f"- coverage_ratio: {float(patched_sample.coverage_ratio):.4f}")
    print(f"- original_full_points: {int(original_full.shape[0])}")
    print(f"- defected_full_points: {int(defected_full.shape[0])}")
    print(f"- patch_centers: {int(centers.shape[0])}")
    print(f"- all_normalized_patch_points: {int(all_norm_defected.shape[0])}")
    print(f"- assembled_world_points: {int(assembled.shape[0])}")
    print(f"- assembled_world_points_dedup: {int(assembled_dedup.shape[0])}")

    clouds = {
        "original_full": original_full,
        "defected_full": defected_full,
        "patch_centers": centers,
        "all_normalized_defected_patches": all_norm_defected,
        "assembled_from_patches": assembled,
        "assembled_deduplicated": assembled_dedup,
    }

    if args.visualize:
        _show_clouds(clouds, point_radius=float(args.point_radius))


if __name__ == "__main__":
    main()
