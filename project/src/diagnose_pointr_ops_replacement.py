"""A/B diagnostics for custom FPS/KNN replacements in PoinTr.

This script checks whether custom operator wrappers behave equivalently to
reference PyTorch3D operations on randomized inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points, sample_farthest_points

from models.pointr.dgcnn import knn_point as dgcnn_knn_point
from models.pointr.transformer import knn_point as transformer_knn_point
from models.pointr.utils import fps, fps_indices, gather_operation


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def _same_knn_sets(idx_a: torch.Tensor, idx_b: torch.Tensor) -> bool:
    """Order-insensitive kNN set comparison per query point."""
    a_sorted = idx_a.sort(dim=-1).values
    b_sorted = idx_b.sort(dim=-1).values
    return bool(torch.equal(a_sorted, b_sorted))


def check_knn_equivalence(device: torch.device) -> CheckResult:
    b, n, s, k = 4, 2048, 256, 16
    xyz = torch.randn(b, n, 3, device=device)
    new_xyz = torch.randn(b, s, 3, device=device)

    idx_ref = knn_points(new_xyz, xyz, K=k, return_nn=False).idx
    idx_dgcnn = dgcnn_knn_point(k, xyz, new_xyz)
    idx_transformer = transformer_knn_point(k, xyz, new_xyz)

    dgcnn_ok = _same_knn_sets(idx_ref, idx_dgcnn)
    transformer_ok = _same_knn_sets(idx_ref, idx_transformer)

    return CheckResult(
        name="kNN set equivalence",
        passed=dgcnn_ok and transformer_ok,
        details=(
            f"dgcnn_match={dgcnn_ok}, transformer_match={transformer_ok}, "
            f"shape_ref={tuple(idx_ref.shape)}"
        ),
    )


def check_fps_points_equivalence(device: torch.device) -> CheckResult:
    b, n, m = 4, 2048, 512
    xyz = torch.randn(b, n, 3, device=device)

    points_ref, _ = sample_farthest_points(xyz, K=m, random_start_point=False)
    points_custom = fps(xyz, m)

    max_abs_diff = float((points_ref - points_custom).abs().max().item())
    passed = max_abs_diff < 1e-6

    return CheckResult(
        name="FPS points equivalence",
        passed=passed,
        details=f"max_abs_diff={max_abs_diff:.8f}, shape={tuple(points_ref.shape)}",
    )


def check_fps_indices_equivalence(device: torch.device) -> CheckResult:
    b, n, m = 4, 2048, 512
    xyz = torch.randn(b, n, 3, device=device)

    _, idx_ref = sample_farthest_points(xyz, K=m, random_start_point=False)
    idx_custom = fps_indices(xyz, m)

    exact_match = bool(torch.equal(idx_ref, idx_custom))
    set_match = _same_knn_sets(idx_ref, idx_custom)

    return CheckResult(
        name="FPS indices equivalence",
        passed=exact_match,
        details=f"exact_match={exact_match}, set_match={set_match}, shape={tuple(idx_ref.shape)}",
    )


def check_gather_equivalence(device: torch.device) -> CheckResult:
    b, c, n, m = 4, 32, 2048, 512
    features = torch.randn(b, c, n, device=device)
    indices = torch.randint(low=0, high=n, size=(b, m), device=device)

    out_custom = gather_operation(features, indices)
    idx_expanded = indices.unsqueeze(1).expand(b, c, m)
    out_ref = torch.gather(features, 2, idx_expanded)

    max_abs_diff = float((out_ref - out_custom).abs().max().item())
    passed = max_abs_diff < 1e-7

    return CheckResult(
        name="gather_operation equivalence",
        passed=passed,
        details=f"max_abs_diff={max_abs_diff:.8f}, shape={tuple(out_ref.shape)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose custom FPS/KNN replacement behavior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    checks = [
        check_knn_equivalence(device),
        check_fps_points_equivalence(device),
        check_fps_indices_equivalence(device),
        check_gather_equivalence(device),
    ]

    print("=== PoinTr Custom Ops Replacement Diagnostics ===")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.details}")

    passed = sum(1 for c in checks if c.passed)
    print(f"Summary: {passed}/{len(checks)} checks passed")


if __name__ == "__main__":
    main()
