from .base import Defect
import numpy as np


class BelowObjectPlane(Defect):
    """
    Adds a synthetic planar artifact beneath an object point cloud.

    This simulates photogrammetry reconstructions where a false floor/plane
    appears under the reconstructed object.
    """

    name: str = "below_object_plane"

    def __init__(
        self,
        num_points: int = 1600,
        offset_ratio: float = 0.05,
        spread_ratio: float = 1.2,
        normal_jitter: float = 0.002,
        plane_jitter: float = 0.01,
        axis: int = 1,
        center_density_bias: float = 0.7,
        edge_sparsity: float = 0.65,
        boundary_irregularity: float = 0.35,
    ):
        """
        Args:
            num_points: Number of points sampled on the synthetic plane.
            offset_ratio: Plane offset under the object along the chosen axis,
                relative to object extent on that axis.
            spread_ratio: Horizontal plane footprint multiplier relative to
                object extent on the two in-plane axes.
            normal_jitter: Small thickness/noise along the plane normal.
            plane_jitter: In-plane random jitter for imperfect sampling.
            axis: Vertical axis index (0=x, 1=y, 2=z). Default is y-up.
            center_density_bias: Higher values produce stronger center density.
            edge_sparsity: Controls how strongly edge points are dropped.
            boundary_irregularity: Controls non-uniform boundary shape.
        """
        self.num_points = int(num_points)
        self.offset_ratio = float(offset_ratio)
        self.spread_ratio = float(spread_ratio)
        self.normal_jitter = float(normal_jitter)
        self.plane_jitter = float(plane_jitter)
        self.axis = int(axis)
        self.center_density_bias = float(center_density_bias)
        self.edge_sparsity = float(edge_sparsity)
        self.boundary_irregularity = float(boundary_irregularity)

    def _sample_irregular_plane_points(
        self,
        center0: float,
        center1: float,
        half0: float,
        half1: float,
        dtype: np.dtype,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample 2D plane coordinates with irregular boundary and center-heavy density."""
        harmonics = np.random.randint(2, 8, size=3)
        phases = np.random.uniform(0.0, 2.0 * np.pi, size=3)
        weights = np.random.uniform(0.35, 1.0, size=3)
        weights = weights / np.sum(weights)

        x_vals: list[np.ndarray] = []
        y_vals: list[np.ndarray] = []
        radii_used: list[np.ndarray] = []
        points_left = self.num_points
        attempts = 0

        # Retry-cap avoids pathological loops when sparsity is very high.
        while points_left > 0 and attempts < 64:
            attempts += 1
            proposal_count = max(points_left * 3, 64)

            theta = np.random.uniform(0.0, 2.0 * np.pi, size=proposal_count)

            # Start from uniform-area disk and bias toward center.
            r = np.sqrt(np.random.uniform(0.0, 1.0, size=proposal_count))
            r = r ** (1.0 + 2.6 * np.clip(self.center_density_bias, 0.0, 1.0))

            # Irregular contour using random low-frequency radial modulation.
            contour = np.zeros_like(theta)
            for h, p, w in zip(harmonics, phases, weights):
                contour += w * np.sin(h * theta + p)
            boundary_scale = (
                1.0 + np.clip(self.boundary_irregularity, 0.0, 1.0) * contour
            )
            boundary_scale = np.clip(boundary_scale, 0.55, 1.45)

            r = r * boundary_scale

            # Keep fewer points near edges.
            norm_r = np.clip(r, 0.0, 1.0)
            keep_prob = 1.0 - np.clip(self.edge_sparsity, 0.0, 0.98) * (norm_r**1.8)
            keep_mask = np.random.uniform(0.0, 1.0, size=proposal_count) < keep_prob

            if not np.any(keep_mask):
                continue

            r_kept = r[keep_mask]
            theta_kept = theta[keep_mask]

            take = min(points_left, r_kept.shape[0])
            r_kept = r_kept[:take]
            theta_kept = theta_kept[:take]

            x_local = r_kept * np.cos(theta_kept) * half0
            y_local = r_kept * np.sin(theta_kept) * half1

            x_vals.append((center0 + x_local).astype(dtype, copy=False))
            y_vals.append((center1 + y_local).astype(dtype, copy=False))
            radii_used.append(r_kept.astype(np.float32, copy=False))
            points_left -= take

        if points_left > 0:
            # Deterministic fallback to guarantee output size.
            theta = np.random.uniform(0.0, 2.0 * np.pi, size=points_left)
            r = np.sqrt(np.random.uniform(0.0, 1.0, size=points_left))
            x_local = r * np.cos(theta) * half0
            y_local = r * np.sin(theta) * half1
            x_vals.append((center0 + x_local).astype(dtype, copy=False))
            y_vals.append((center1 + y_local).astype(dtype, copy=False))
            radii_used.append(r.astype(np.float32, copy=False))

        return (
            np.concatenate(x_vals),
            np.concatenate(y_vals),
            np.concatenate(radii_used),
        )

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0 or self.num_points <= 0:
            return points, {"points_added": 0}

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points shape (N, 3), got {points.shape}")
        if self.axis < 0 or self.axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2; got {self.axis}")

        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (min_xyz + max_xyz) * 0.5
        extent = np.maximum(max_xyz - min_xyz, 1e-6)

        plane_coord = min_xyz[self.axis] - self.offset_ratio * extent[self.axis]

        in_plane_axes = [a for a in range(3) if a != self.axis]
        ax0, ax1 = in_plane_axes

        half0 = 0.5 * extent[ax0] * self.spread_ratio
        half1 = 0.5 * extent[ax1] * self.spread_ratio

        axis0_vals, axis1_vals, sampled_r = self._sample_irregular_plane_points(
            center0=center[ax0],
            center1=center[ax1],
            half0=half0,
            half1=half1,
            dtype=points.dtype,
        )

        plane_points = np.zeros((self.num_points, 3), dtype=points.dtype)
        plane_points[:, ax0] = axis0_vals
        plane_points[:, ax1] = axis1_vals
        plane_points[:, self.axis] = plane_coord

        if self.plane_jitter > 0:
            # Slightly more in-plane jitter near boundaries to avoid crisp edges.
            jitter_scale = self.plane_jitter * (
                0.35 + 0.9 * np.clip(sampled_r, 0.0, 1.5)
            )
            plane_points[:, ax0] += np.random.normal(
                0.0,
                jitter_scale,
            )
            plane_points[:, ax1] += np.random.normal(
                0.0,
                jitter_scale,
            )

        if self.normal_jitter > 0:
            plane_points[:, self.axis] += np.random.normal(
                0.0,
                self.normal_jitter,
                size=self.num_points,
            )

        new_points = np.concatenate([points, plane_points], axis=0)

        metadata = {
            "points_added": int(self.num_points),
            "axis": int(self.axis),
            "offset_ratio": float(self.offset_ratio),
            "spread_ratio": float(self.spread_ratio),
            "normal_jitter": float(self.normal_jitter),
            "plane_jitter": float(self.plane_jitter),
            "center_density_bias": float(self.center_density_bias),
            "edge_sparsity": float(self.edge_sparsity),
            "boundary_irregularity": float(self.boundary_irregularity),
            "plane_coord": float(plane_coord),
            "sampled_radius_mean": float(np.mean(sampled_r)),
        }

        return new_points, metadata
