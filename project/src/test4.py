# Simple .ply loader and polyscope visualizer
import sys
from pathlib import Path
import numpy as np
import trimesh
import polyscope as ps


def main():
    if len(sys.argv) > 1:
        ply_path = Path(sys.argv[1])
    else:
        ply_path = Path("./duster_output.ply")
    assert ply_path.exists(), f"PLY file {ply_path} does not exist!"

    # Load point cloud
    print(f"Loading point cloud from {ply_path}")
    pc = trimesh.load(str(ply_path))
    pts = np.asarray(pc.vertices)
    colors = (
        np.asarray(pc.colors)
        if hasattr(pc, "colors") and pc.colors is not None
        else None
    )
    # If colors are RGBA, use only RGB
    if colors is not None and colors.shape[1] == 4:
        colors = colors[:, :3]

    # Visualize with polyscope
    print("Visualizing point cloud with polyscope...")
    ps.init()
    ps.register_point_cloud("Loaded Point Cloud", pts, enabled=True)
    if colors is not None:
        ps.get_point_cloud("Loaded Point Cloud").add_color_quantity(
            "rgb", colors, enabled=True
        )
    ps.show()


if __name__ == "__main__":
    main()


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
