from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load NPZ point cloud and visualize it with Polyscope."
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        default="./test.npz",
        help="Path to NPZ file containing point cloud arrays.",
    )
    parser.add_argument(
        "--input-key",
        type=str,
        default="completed_pos",
        help="NPZ key to visualize.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all compatible point clouds from NPZ (N,3) in one Polyscope scene.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="npz_pointcloud",
        help="Polyscope object name.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0022,
        help="Point radius in Polyscope.",
    )
    return parser


def _to_points(array: np.ndarray) -> np.ndarray:
    points = np.asarray(array, dtype=np.float32)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected point cloud with shape (N, 3), got {tuple(points.shape)}"
        )

    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or Inf values.")

    return points


def main() -> None:
    args = _build_parser().parse_args()

    input_npz = Path(args.input_npz)
    if not input_npz.exists():
        raise FileNotFoundError(f"NPZ file not found: {input_npz}")

    npz = np.load(str(input_npz))
    keys = list(npz.keys())
    if not keys:
        raise ValueError(f"NPZ file contains no arrays: {input_npz}")

    import polyscope as ps

    ps.init()

    print(f"[INFO] Loaded: {input_npz}")
    if bool(args.show_all):
        shown = 0
        for key in keys:
            try:
                points = _to_points(npz[key])
            except ValueError:
                continue

            ps.register_point_cloud(
                key,
                points,
                point_render_mode="quad",
                radius=float(args.radius),
            )
            print(f"[INFO] Key: {key}, shape: {points.shape}, dtype: {points.dtype}")
            shown += 1

        if shown == 0:
            raise ValueError(
                "No compatible point cloud arrays with shape (N, 3) were found in NPZ."
            )
    else:
        selected_key = str(args.input_key)
        if selected_key not in npz:
            raise KeyError(
                f"Key '{selected_key}' not found in {input_npz}. Available keys: {keys}"
            )

        points = _to_points(npz[selected_key])
        print(f"[INFO] Key: {selected_key}")
        print(f"[INFO] Shape: {points.shape}, dtype: {points.dtype}")
        ps.register_point_cloud(
            str(args.name),
            points,
            point_render_mode="quad",
            radius=float(args.radius),
        )

    ps.show()


if __name__ == "__main__":
    main()
