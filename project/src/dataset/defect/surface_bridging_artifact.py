from .base import Defect
import numpy as np


class SurfaceBridgingArtifact(Defect):
    """
    Creates a thin surface-like synthetic bridge between two distant regions
    of the point cloud.

    Purpose:
        - Simulates MVS hallucinated surfaces that incorrectly connect
          object parts or background fragments.
        - Produces sheet-like artifacts instead of lines, matching real
          photogrammetry behavior (ghost walls, flat patches, false plates).
        - Useful for training the model to remove thin invalid geometry
          and preserve correct topology.

    Parameters:
        num_surfaces (int):
            Number of surfaces (bridges) to generate.

        resolution_u (int):
            Number of samples along the main A->B axis.

        resolution_v (int):
            Number of samples along the perpendicular axis (surface width).

        width (float):
            Half-width of the surface (thickness perpendicular to A->B axis).

        jitter (float):
            Noise applied to surface points to avoid perfect planar shape.
    """

    name: str = "surface_bridging_artifact"

    def __init__(
        self,
        num_surfaces: int = 1,
        resolution_u: int = 30,
        resolution_v: int = 6,
        width: float = 0.02,
        jitter: float = 0.005,
    ):
        self.num_surfaces = int(num_surfaces)
        self.resolution_u = int(resolution_u)
        self.resolution_v = int(resolution_v)
        self.width = float(width)
        self.jitter = float(jitter)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0:
            return points, {"surfaces_used": 0}

        # Bounding box
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        bbox_center = (min_xyz + max_xyz) * 0.5
        bbox_extent = np.linalg.norm(max_xyz - min_xyz)

        surfaces = []

        for _ in range(self.num_surfaces):
            # Pick one endpoint A from point cloud
            A = points[np.random.randint(points.shape[0])]

            # Opposite endpoint B far away from A
            direction = bbox_center - A
            direction /= np.linalg.norm(direction) + 1e-8
            B = bbox_center + direction * bbox_extent

            # Main axis vector
            AB = B - A
            AB_norm = AB / (np.linalg.norm(AB) + 1e-8)

            # Build perpendicular basis (u, v)
            # u = perpendicular to AB
            rand = np.random.normal(0.0, 1.0, size=3)
            u = np.cross(AB_norm, rand)
            u /= np.linalg.norm(u) + 1e-8

            # v = perpendicular to AB and u
            v = np.cross(AB_norm, u)
            v /= np.linalg.norm(v) + 1e-8

            # Parametric grid for the surface
            t_u = np.linspace(0.0, 1.0, self.resolution_u)
            t_v = np.linspace(-self.width, self.width, self.resolution_v)

            surface_points = []

            for tu in t_u:
                # Interpolate along the AB axis
                p = A * (1 - tu) + B * tu
                for tv in t_v:
                    # Create offset using u and v directions
                    offset = tv * u
                    surface_points.append(p + offset)

            surface = np.array(surface_points)

            # Add jitter noise for realism
            if self.jitter > 0:
                surface += np.random.normal(0.0, self.jitter, size=surface.shape)

            surfaces.append(surface)

        new_points = np.concatenate([points] + surfaces, axis=0)

        metadata = {
            "surfaces_used": self.num_surfaces,
            "resolution_u": self.resolution_u,
            "resolution_v": self.resolution_v,
            "width": self.width,
            "jitter": self.jitter,
        }

        return new_points, metadata
