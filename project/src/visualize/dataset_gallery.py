from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import textwrap

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
import torch


@dataclass
class GalleryConfig:
    views: Sequence[Tuple[float, float]] = ((18.0, 35.0), (18.0, 125.0), (18.0, 215.0))
    point_size: float = 6.0
    max_points: int = 12000
    dpi: int = 220
    page_size: Optional[Tuple[float, float]] = None
    background_color: str = "#ffffff"
    point_cmap: str = "gnuplot"
    zoom: float = 1.0
    max_sample_cols: int = 3
    caption_fontsize: int = 6
    title_fontsize: int = 8
    wrap_caption_chars: int = 100
    sample_title_fontsize: int = 7
    badge_fontsize: int = 5
    border_color: str = "#9ca3af"
    border_linewidth: float = 0.8
    description_color: str = "#6b7280"
    outer_wspace: float = 0.1
    outer_hspace: float = 0.1
    inner_wspace: float = 0
    inner_hspace: float = 0
    block_view_width: float = 1
    block_row_height: float = 1
    min_figure_width: float = 0
    min_figure_height: float = 0
    image_grid_cols: int = 0
    image_grid_max_images: int = 0
    image_row_height_ratio: float = 1.8
    side_note_fontsize: int = 5


def _to_numpy_points(obj: Any) -> Optional[np.ndarray]:
    if obj is None:
        return None

    if hasattr(obj, "vertices"):
        arr = np.asarray(obj.vertices)
    elif isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif hasattr(obj, "pos"):
        pos = obj.pos
        arr = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
    else:
        arr = np.asarray(obj)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2 or arr.shape[1] != 3:
        return None

    return arr.astype(np.float32, copy=False)


def _extract_points_and_colors(obj: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract points and optional RGB colors from supported point-cloud containers."""
    points = _to_numpy_points(obj)
    if points is None:
        return None, None

    colors = None
    if hasattr(obj, "colors"):
        raw_colors = np.asarray(obj.colors)
        if raw_colors.ndim == 2 and raw_colors.shape[0] == points.shape[0]:
            if raw_colors.shape[1] >= 3:
                colors = raw_colors[:, :3].astype(np.float32, copy=False)

    if colors is not None and colors.size > 0:
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    return points, colors

def _normalize_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    scale = max(float(scale), 1e-8)
    return points / scale


def _subsample_points(
    points: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.shape[0] <= max_points:
        return points, colors
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    points = points[idx]
    if colors is not None:
        colors = colors[idx]
    return points, colors


def render_pointcloud_view(
    cloud: Any,
    elev: float,
    azim: float,
    *,
    point_size: float,
    background_color: str,
    point_cmap: str,
    rng: np.random.Generator,
    max_points: int,
    zoom: float,
) -> np.ndarray:
    """Render one point cloud view to an RGB image array."""
    points, colors = _extract_points_and_colors(cloud)
    if points is None:
        raise ValueError("Could not extract point cloud points for rendering")

    points = _normalize_points(points)
    pts, colors = _subsample_points(points, max_points=max_points, rng=rng, colors=colors)

    fig = plt.figure(figsize=(3.2, 3.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    if colors is not None:
        # Dense photogrammetric clouds are easier to read with original RGB colors.
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=point_size,
            c=colors,
            marker=".",
            linewidths=0,
            alpha=1.0,
            depthshade=False,
        )
    else:
        color_values = pts[:, 2]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=point_size,
            c=color_values,
            cmap=point_cmap,
            marker="o",
            linewidths=0,
            alpha=0.95,
            depthshade=False,
        )

    effective_zoom = max(0.1, float(zoom))
    lim = 0.9 / effective_zoom
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return img


def render_source_image_grid(
    paths: Sequence[str],
    *,
    background_color: str,
    grid_cols: int = 0,
    max_images: int = 0,
) -> np.ndarray:
    """Render all source images as a wrapped grid fitted into one row panel."""
    if max_images > 0:
        paths = list(paths)[:max_images]

    n = len(paths)
    if n == 0:
        fig = plt.figure(figsize=(3.2, 1.8), dpi=220)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.text(0.5, 0.5, "No images", ha="center", va="center", color="#374151", fontsize=9)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        return out

    cols = grid_cols if grid_cols and grid_cols > 0 else int(np.ceil(np.sqrt(n)))
    cols = max(1, cols)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(max(0, cols * 2), max(0, rows * 2)), dpi=200)
    fig.patch.set_facecolor(background_color)
    grid = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.1)

    for i in range(rows * cols):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        ax.set_facecolor(background_color)
        ax.set_axis_off()

        if i >= n:
            continue

        img = plt.imread(paths[i])
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

        ax.imshow(img)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return out


def _rounded_rect_path(x: float, y: float, w: float, h: float, r: float) -> mpath.Path:
    # Build rounded rectangle in axes coordinates using cubic Bezier corners.
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    k = 0.5522847498307936  # Circle approximation constant for cubic Bezier.
    c = r * k

    verts = [
        (x + r, y),
        (x + w - r, y),
        (x + w - r + c, y),
        (x + w, y + r - c),
        (x + w, y + r),
        (x + w, y + h - r),
        (x + w, y + h - r + c),
        (x + w - r + c, y + h),
        (x + w - r, y + h),
        (x + r, y + h),
        (x + r - c, y + h),
        (x, y + h - r + c),
        (x, y + h - r),
        (x, y + r),
        (x, y + r - c),
        (x + r - c, y),
        (x + r, y),
        (x + r, y),
    ]

    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CLOSEPOLY,
    ]

    return mpath.Path(verts, codes)


def create_dataset_gallery_figure(
    pointclouds: Sequence[Any],
    *,
    dataset_name: str,
    sample_indices: Sequence[int],
    descriptions: Optional[Sequence[str]] = None,
    badge_labels: Optional[Sequence[Sequence[str]]] = None,
    badge_details: Optional[Sequence[Sequence[str]]] = None,
    side_notes: Optional[Sequence[str]] = None,
    config: Optional[GalleryConfig] = None,
    seed: int = 42,
):
    """Create a publication-ready multi-sample, multi-view dataset gallery figure."""
    if config is None:
        config = GalleryConfig()

    if len(pointclouds) == 0:
        raise ValueError("No point clouds provided")

    rng = np.random.default_rng(seed)
    views = list(config.views)
    n_samples = len(pointclouds)
    cloud_rows_per_sample = [len(pc) if isinstance(pc, (tuple, list)) else 1 for pc in pointclouds]
    max_cloud_rows = max(cloud_rows_per_sample)

    n_sample_cols = max(1, min(config.max_sample_cols, n_samples))
    n_sample_rows = int(np.ceil(n_samples / n_sample_cols))

    sample_row_count = max_cloud_rows
    if config.page_size is None:
        sample_block_w = len(views) * config.block_view_width
        sample_block_h = sample_row_count * config.block_row_height
        fig_w = max(config.min_figure_width, n_sample_cols * sample_block_w)
        fig_h = max(config.min_figure_height, n_sample_rows * sample_block_h)
        page_size = (fig_w, fig_h)
    else:
        page_size = config.page_size

    fig = plt.figure(figsize=page_size, dpi=config.dpi)
    fig.patch.set_facecolor("white")

    outer = fig.add_gridspec(
        n_sample_rows,
        n_sample_cols,
        left=0.06,
        right=0.94,
        top=0.89,
        bottom=0.10,
        wspace=config.outer_wspace,
        hspace=config.outer_hspace,
    )

    for sample_idx, point_entry in enumerate(pointclouds):
        sr = sample_idx // n_sample_cols
        sc = sample_idx % n_sample_cols

        if isinstance(point_entry, (tuple, list)):
            clouds = list(point_entry)
        else:
            clouds = [point_entry]

        row_count = max_cloud_rows

        if badge_labels is not None and sample_idx < len(badge_labels):
            row_badges = list(badge_labels[sample_idx])
        elif len(clouds) == 2:
            row_badges = ["Original", "Defected"]
        else:
            row_badges = [f"Cloud {i + 1}" for i in range(len(clouds))]

        if len(row_badges) < len(clouds):
            for i in range(len(row_badges), len(clouds)):
                row_badges.append(f"Cloud {i + 1}")

        row_details: List[str] = []
        if badge_details is not None and sample_idx < len(badge_details):
            row_details = [d or "" for d in badge_details[sample_idx]]
        if len(row_details) < len(clouds):
            row_details.extend([""] * (len(clouds) - len(row_details)))

        inner = outer[sr, sc].subgridspec(
            row_count,
            len(views),
            height_ratios=[
                config.image_row_height_ratio
                if (i < len(clouds) and isinstance(clouds[i], dict) and clouds[i].get("type") in {"images", "masks"})
                else 1.0
                for i in range(row_count)
            ],
            wspace=config.inner_wspace,
            hspace=config.inner_hspace,
        )

        for group_idx in range(row_count):
            if group_idx >= len(clouds):
                for view_idx in range(len(views)):
                    ax = fig.add_subplot(inner[group_idx, view_idx])
                    ax.axis("off")
                continue

            row_item = clouds[group_idx]
            is_image_row = isinstance(row_item, dict) and row_item.get("type") in {"images", "masks"}

            if is_image_row:
                ax = fig.add_subplot(inner[group_idx, :])
                img = render_source_image_grid(
                    row_item.get("paths", []),
                    background_color=config.background_color,
                    grid_cols=config.image_grid_cols,
                    max_images=config.image_grid_max_images,
                )
                ax.imshow(img)
                ax.set_axis_off()
            else:
                for view_idx, (elev, azim) in enumerate(views):
                    ax = fig.add_subplot(inner[group_idx, view_idx])
                    img = render_pointcloud_view(
                        row_item,
                        elev,
                        azim,
                        point_size=config.point_size,
                        background_color=config.background_color,
                        point_cmap=config.point_cmap,
                        rng=rng,
                        max_points=config.max_points,
                        zoom=config.zoom,
                    )
                    ax.imshow(img)
                    ax.set_axis_off()

            badge_text = row_badges[group_idx]
            low = badge_text.lower()
            if low.startswith("orig") or low == "gt":
                badge_color = "#dbeafe"
            elif "defect" in low:
                badge_color = "#fee2e2"
            elif "image" in low or "frame" in low:
                badge_color = "#fef9c3"
            elif "mask" in low:
                badge_color = "#e5e7eb"
            else:
                badge_color = "#ecfeff"

            badge_ax = fig.add_subplot(inner[group_idx, 0])
            badge_ax.text(
                0.05,
                0.9,
                badge_text,
                transform=badge_ax.transAxes,
                ha="left",
                va="top",
                fontsize=config.badge_fontsize,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": badge_color,
                    "edgecolor": "#cbd5e1",
                    "linewidth": 0.8,
                },
            )

            detail_text = row_details[group_idx]
            if detail_text:
                badge_ax.text(
                    0.05,
                    0.70,
                    detail_text,
                    transform=badge_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(4, config.badge_fontsize - 1),
                    color="#111827",
                    linespacing=1.2,
                )
            badge_ax.set_axis_off()

        border_ax = fig.add_subplot(outer[sr, sc])
        border_ax.set_xticks([])
        border_ax.set_yticks([])
        border_ax.patch.set_alpha(0.0)
        for spine in border_ax.spines.values():
            spine.set_visible(False)

        border_ax.add_patch(
            mpatches.PathPatch(
                _rounded_rect_path(0, 0, 1, 1, 0),
                transform=border_ax.transAxes,
                fill=False,
                edgecolor=config.border_color,
                linewidth=config.border_linewidth,
                capstyle="round",
                joinstyle="round",
                clip_on=False,
            )
        )

        desc = ""
        if descriptions is not None and sample_idx < len(descriptions):
            desc = descriptions[sample_idx] or ""
        wrapped_desc = textwrap.fill(desc, width=config.wrap_caption_chars)
        border_ax.text(
            0.0,
            -0.085,
            wrapped_desc,
            transform=border_ax.transAxes,
            ha="left",
            va="top",
            fontsize=config.caption_fontsize,
            color=config.description_color,
            clip_on=False,
        )

        side_note = ""
        if side_notes is not None and sample_idx < len(side_notes):
            side_note = side_notes[sample_idx] or ""
        if side_note:
            border_ax.text(
                1.02,
                0.5,
                side_note,
                transform=border_ax.transAxes,
                ha="left",
                va="center",
                fontsize=config.side_note_fontsize,
                color="#1f2937",
                clip_on=False,
                linespacing=1.25,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#f8fafc",
                    "edgecolor": "#cbd5e1",
                    "linewidth": 0.7,
                },
            )
    return fig


def save_dataset_gallery(
    pointclouds: Sequence[Any],
    output_path: str,
    *,
    dataset_name: str,
    sample_indices: Sequence[int],
    descriptions: Optional[Sequence[str]] = None,
    badge_labels: Optional[Sequence[Sequence[str]]] = None,
    badge_details: Optional[Sequence[Sequence[str]]] = None,
    side_notes: Optional[Sequence[str]] = None,
    config: Optional[GalleryConfig] = None,
    seed: int = 42,
):
    """Create and save a dataset gallery image."""
    fig = create_dataset_gallery_figure(
        pointclouds,
        dataset_name=dataset_name,
        sample_indices=sample_indices,
        descriptions=descriptions,
        badge_labels=badge_labels,
        badge_details=badge_details,
        side_notes=side_notes,
        config=config,
        seed=seed,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_defect_log(defect_log: Dict[str, Dict[str, Any]]) -> str:
    if not defect_log:
        return "no defects logged"

    chunks = []
    for defect_name, params in defect_log.items():
        if not params:
            chunks.append(defect_name)
            continue

        param_parts = []
        for key, value in params.items():
            if isinstance(value, float):
                param_parts.append(f"{key}={value:.3f}")
            else:
                param_parts.append(f"{key}={value}")
        chunks.append(f"{defect_name}({', '.join(param_parts)})")

    return " | ".join(chunks)
