from .base import Defect
import numpy as np


class HairLikeNoise(Defect):
    """
    HairLikeNoise adds spike-like protrusions to random points in the cloud,
    simulating needle-shaped artifacts commonly seen in MVS depth fusion.

    Purpose:
        - Models photogrammetry errors where points shoot outward from silhouette
          edges or discontinuities.
        - Trains the model to suppress invalid elongated noise.
        - A realistic complement to both Gaussian noise and bridging artifacts.

    Parameters:
        num_spikes (int):
            Number of spike artifacts to generate.

        max_length (float):
            Maximum length of each spike.

        jitter (float):
            Small noise added along the spike for realism.
    """

    name: str = "hair_like_noise"

    def __init__(
        self,
        num_spikes: int = 50,
        max_length: float = 0.05,
        jitter: float = 0.003,
    ):
        self.num_spikes = int(num_spikes)
        self.max_length = float(max_length)
        self.jitter = float(jitter)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        if points.size == 0:
            return points, {"spikes_used": 0}

        n = points.shape[0]

        # Randomly pick spike origins
        idxs = np.random.choice(n, size=min(self.num_spikes, n), replace=False)
        origins = points[idxs]

        spikes = []

        for origin in origins:
            # Random direction, normalized
            direction = np.random.normal(0.0, 1.0, size=3)
            direction /= np.linalg.norm(direction) + 1e-8

            # Spike length
            length = np.random.uniform(0.2, 1.0) * self.max_length

            # Create the spike (line samples)
            t = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
            spike_points = origin + direction * (t * length)

            # Jitter for realism
            if self.jitter > 0:
                spike_points += np.random.normal(
                    0.0, self.jitter, size=spike_points.shape
                )

            spikes.append(spike_points)

        new_points = np.concatenate([points] + spikes, axis=0)

        metadata = {
            "spikes_used": len(spikes),
            "max_length": self.max_length,
            "jitter": self.jitter,
        }

        return new_points, metadata
