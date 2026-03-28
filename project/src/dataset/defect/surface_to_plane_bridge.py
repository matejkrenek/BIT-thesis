from .base import Defect
import numpy as np


class SurfaceToPlaneBridge(Defect):
    """
    Adds thin bridge-like artifacts from the object surface toward a synthetic
    plane below the object.

    This mimics reconstruction failures where geometry gets incorrectly
    connected to the support surface or floor.
    """

    name: str = "surface_to_plane_bridge"

    def __init__(
        self,
        num_bridges: int = 8,
        points_per_bridge: int = 18,
        plane_offset_ratio: float = 0.04,
        axis: int = 1,
        bottom_band_ratio: float = 0.35,
        top_band_ratio: float = 0.25,
        side_band_ratio: float = 0.22,
        bottom_bridge_fraction: float = 0.3,
        top_bridge_fraction: float = 0.25,
        side_bridge_fraction: float = 0.45,
        diagonal_strength_min: float = 0.15,
        diagonal_strength_max: float = 0.55,
        lateral_jitter: float = 0.003,
        normal_jitter: float = 0.0015,
    ):
        self.num_bridges = int(num_bridges)
        self.points_per_bridge = int(points_per_bridge)
        self.plane_offset_ratio = float(plane_offset_ratio)
        self.axis = int(axis)
        self.bottom_band_ratio = float(bottom_band_ratio)
        self.top_band_ratio = float(top_band_ratio)
        self.side_band_ratio = float(side_band_ratio)
        self.bottom_bridge_fraction = float(bottom_bridge_fraction)
        self.top_bridge_fraction = float(top_bridge_fraction)
        self.side_bridge_fraction = float(side_bridge_fraction)
        self.diagonal_strength_min = float(diagonal_strength_min)
        self.diagonal_strength_max = float(diagonal_strength_max)
        self.lateral_jitter = float(lateral_jitter)
        self.normal_jitter = float(normal_jitter)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0 or self.num_bridges <= 0 or self.points_per_bridge <= 1:
            return points, {"bridges_used": 0}

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points shape (N, 3), got {points.shape}")
        if self.axis < 0 or self.axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2; got {self.axis}")

        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (min_xyz + max_xyz) * 0.5
        extent = np.maximum(max_xyz - min_xyz, 1e-6)

        plane_coord = min_xyz[self.axis] - self.plane_offset_ratio * extent[self.axis]

        in_plane_axes = [a for a in range(3) if a != self.axis]
        ax0, ax1 = in_plane_axes

        # Partition points into likely source zones for realistic bridge starts.
        normalized_h = (points[:, self.axis] - min_xyz[self.axis]) / extent[self.axis]
        bottom_mask = normalized_h <= np.clip(self.bottom_band_ratio, 0.0, 1.0)
        top_mask = normalized_h >= (1.0 - np.clip(self.top_band_ratio, 0.0, 1.0))

        rel0 = np.abs(points[:, ax0] - center[ax0]) / (0.5 * extent[ax0] + 1e-8)
        rel1 = np.abs(points[:, ax1] - center[ax1]) / (0.5 * extent[ax1] + 1e-8)
        side_threshold = 1.0 - np.clip(self.side_band_ratio, 0.0, 0.95)
        side_mask = np.maximum(rel0, rel1) >= side_threshold

        zone_candidates = {
            "bottom": points[bottom_mask],
            "top": points[top_mask],
            "side": points[side_mask],
        }

        zone_weights = np.array(
            [
                max(self.bottom_bridge_fraction, 0.0),
                max(self.top_bridge_fraction, 0.0),
                max(self.side_bridge_fraction, 0.0),
            ],
            dtype=np.float64,
        )
        if np.sum(zone_weights) <= 0:
            zone_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        zone_names = ["bottom", "top", "side"]
        zone_probs = zone_weights / np.sum(zone_weights)

        zone_use_counts = {"bottom": 0, "top": 0, "side": 0, "fallback": 0}

        bridges = []
        for _ in range(self.num_bridges):
            chosen_zone = np.random.choice(zone_names, p=zone_probs)
            candidates = zone_candidates[chosen_zone]

            if candidates.shape[0] == 0:
                candidates = points
                zone_use_counts["fallback"] += 1
            else:
                zone_use_counts[chosen_zone] += 1

            start = candidates[np.random.randint(candidates.shape[0])].copy()
            end = start.copy()
            end[self.axis] = plane_coord

            # Force diagonal descent by shifting endpoint laterally in a random in-plane direction.
            descent = max(start[self.axis] - plane_coord, 0.0)
            direction_2d = np.random.normal(0.0, 1.0, size=2)
            direction_norm = np.linalg.norm(direction_2d)
            if direction_norm < 1e-8:
                direction_2d = np.array([1.0, 0.0], dtype=np.float64)
                direction_norm = 1.0
            direction_2d = direction_2d / direction_norm

            diagonal_min = max(self.diagonal_strength_min, 0.0)
            diagonal_max = max(self.diagonal_strength_max, diagonal_min)
            diagonal_scale = np.random.uniform(diagonal_min, diagonal_max)
            lateral_shift = descent * diagonal_scale

            end[ax0] += direction_2d[0] * lateral_shift
            end[ax1] += direction_2d[1] * lateral_shift
            end[ax0] += np.random.normal(0.0, self.lateral_jitter * 2.0)
            end[ax1] += np.random.normal(0.0, self.lateral_jitter * 2.0)

            t = np.linspace(0.0, 1.0, self.points_per_bridge, dtype=np.float32)[:, None]
            bridge = start[None, :] * (1.0 - t) + end[None, :] * t

            if self.lateral_jitter > 0:
                bridge[:, ax0] += np.random.normal(
                    0.0, self.lateral_jitter, size=self.points_per_bridge
                )
                bridge[:, ax1] += np.random.normal(
                    0.0, self.lateral_jitter, size=self.points_per_bridge
                )

            if self.normal_jitter > 0:
                bridge[:, self.axis] += np.random.normal(
                    0.0, self.normal_jitter, size=self.points_per_bridge
                )

            bridges.append(bridge.astype(points.dtype, copy=False))

        new_points = np.concatenate([points] + bridges, axis=0)

        metadata = {
            "bridges_used": int(self.num_bridges),
            "points_per_bridge": int(self.points_per_bridge),
            "axis": int(self.axis),
            "plane_offset_ratio": float(self.plane_offset_ratio),
            "bottom_band_ratio": float(self.bottom_band_ratio),
            "top_band_ratio": float(self.top_band_ratio),
            "side_band_ratio": float(self.side_band_ratio),
            "bottom_bridge_fraction": float(self.bottom_bridge_fraction),
            "top_bridge_fraction": float(self.top_bridge_fraction),
            "side_bridge_fraction": float(self.side_bridge_fraction),
            "diagonal_strength_min": float(self.diagonal_strength_min),
            "diagonal_strength_max": float(self.diagonal_strength_max),
            "plane_coord": float(plane_coord),
            "lateral_jitter": float(self.lateral_jitter),
            "normal_jitter": float(self.normal_jitter),
            "zone_use_counts": zone_use_counts,
        }

        return new_points, metadata
