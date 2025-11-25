from .base import Defect
import numpy as np


class BridgingArtifact(Defect):
    """
    Creates artificial bridges between distant parts of the object by selecting
    endpoints on opposite sides of the bounding box. This guarantees visible,
    structural bridging even for single solid objects.

    Purpose:
        - Simulates incorrect MVS connectivity between regions that should not
          be connected (e.g., object ↔ background, two legs of a chair, etc.).
        - Always produces visible artifacts, even on single-piece models.
    """

    name: str = "bridging_artifact"

    def __init__(
        self,
        num_bridges: int = 1,
        points_per_bridge: int = 40,
        jitter: float = 0.01,
    ):
        self.num_bridges = int(num_bridges)
        self.points_per_bridge = int(points_per_bridge)
        self.jitter = float(jitter)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0:
            return points, {"bridges_used": 0}

        # Bounding box
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        bbox_center = (min_xyz + max_xyz) * 0.5

        bridges = []

        for _ in range(self.num_bridges):
            # First endpoint = real point
            A = points[np.random.randint(points.shape[0])]

            # Second endpoint = opposite side of bounding box
            direction = bbox_center - A
            direction /= np.linalg.norm(direction) + 1e-8
            B = bbox_center + direction * np.linalg.norm(max_xyz - min_xyz)

            # Create bridge points
            t = np.linspace(0.0, 1.0, self.points_per_bridge).reshape(-1, 1)
            bridge = A * (1 - t) + B * t

            # Add slight noise
            bridge += np.random.normal(0.0, self.jitter, size=bridge.shape)

            bridges.append(bridge)

        new_points = np.concatenate([points] + bridges, axis=0)

        metadata = {
            "bridges_used": self.num_bridges,
            "points_per_bridge": self.points_per_bridge,
            "jitter": self.jitter,
        }

        return new_points, metadata
