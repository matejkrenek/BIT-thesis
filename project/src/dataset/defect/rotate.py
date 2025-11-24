from .base import Defect
import numpy as np


class Rotate(Defect):
    name: str = "rotate"

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x_rad = np.radians(x)
        self.y_rad = np.radians(y)
        self.z_rad = np.radians(z)

    def apply(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.x_rad), -np.sin(self.x_rad)],
                [0, np.sin(self.x_rad), np.cos(self.x_rad)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(self.y_rad), 0, np.sin(self.y_rad)],
                [0, 1, 0],
                [-np.sin(self.y_rad), 0, np.cos(self.y_rad)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(self.z_rad), -np.sin(self.z_rad), 0],
                [np.sin(self.z_rad), np.cos(self.z_rad), 0],
                [0, 0, 1],
            ]
        )
        R = Rz @ Ry @ Rx

        rotated = points @ R.T

        return rotated, {
            "x_deg": np.degrees(self.x_rad),
            "y_deg": np.degrees(self.y_rad),
            "z_deg": np.degrees(self.z_rad),
        }
