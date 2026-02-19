import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def plot_pointcloud_to_image(
    pointcloud,
    output_path=None,
    figsize=(6, 6),
    dpi=100,
    elev=20,
    azim=45,
    point_size=1.0,
):
    if isinstance(pointcloud, torch.Tensor):
        pointcloud = pointcloud.detach().cpu().numpy()

    pointcloud = np.asarray(pointcloud)

    if pointcloud.ndim == 3 and pointcloud.shape[0] == 1:
        pointcloud = pointcloud[0]

    if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
        raise ValueError("pointcloud must have shape (N, 3) or (1, N, 3)")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pointcloud[:, 0],
        pointcloud[:, 1],
        pointcloud[:, 2],
        s=point_size,
        c=pointcloud[:, 2],
        cmap="viridis",
        linewidths=0,
    )

    mins = pointcloud.min(axis=0)
    maxs = pointcloud.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(pad=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    return image


def plot_dense_pointcloud_to_image(
    pointcloud,
    colors=None,
    output_path=None,
    figsize=(6, 6),
    dpi=150,
    elev=20,
    azim=45,
    point_size=0.25,
    max_points=250000,
    seed=42,
):
    if hasattr(pointcloud, "vertices"):
        vertices = np.asarray(pointcloud.vertices)
        if colors is None and hasattr(pointcloud, "colors"):
            colors = np.asarray(pointcloud.colors)
        pointcloud = vertices

    if isinstance(pointcloud, torch.Tensor):
        pointcloud = pointcloud.detach().cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    pointcloud = np.asarray(pointcloud)
    if colors is not None:
        colors = np.asarray(colors)

    if pointcloud.ndim == 3 and pointcloud.shape[0] == 1:
        pointcloud = pointcloud[0]
    if colors is not None and colors.ndim == 3 and colors.shape[0] == 1:
        colors = colors[0]

    if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
        raise ValueError("pointcloud must have shape (N, 3) or (1, N, 3)")
    if colors is None:
        colors = np.ones((pointcloud.shape[0], 3), dtype=np.float32)
    else:
        if colors.ndim != 2 or colors.shape[1] not in (3, 4):
            raise ValueError("colors must have shape (N, 3), (N, 4), (1, N, 3), or (1, N, 4)")
        if pointcloud.shape[0] != colors.shape[0]:
            raise ValueError("pointcloud and colors must have the same number of points")
        if colors.shape[1] == 4:
            colors = colors[:, :3]

    num_points = pointcloud.shape[0]
    if num_points > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(num_points, size=max_points, replace=False)
        pointcloud = pointcloud[idx]
        colors = colors[idx]

    colors = colors.astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pointcloud[:, 0],
        pointcloud[:, 1],
        pointcloud[:, 2],
        s=point_size,
        c=colors,
        marker=".",
        linewidths=0,
        depthshade=False,
    )

    mins = pointcloud.min(axis=0)
    maxs = pointcloud.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(pad=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    return image


plot_toponmtincloud_to_image = plot_dense_pointcloud_to_image