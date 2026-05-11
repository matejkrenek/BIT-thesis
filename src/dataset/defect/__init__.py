"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: __init__.py
Responsibility: Exports synthetic defect classes for point cloud corruption and augmentation.
"""

from .base import Defect
from .noise import Noise
from .local_dropout import LocalDropout
from .outlier_points import OutlierPoints
from .large_missing_region import LargeMissingRegion
from .below_object_plane import BelowObjectPlane
from .surface_to_plane_bridge import SurfaceToPlaneBridge
from .combined import Combined

__all__ = [
    "Defect",
    "Noise",
    "LocalDropout",
    "OutlierPoints",
    "LargeMissingRegion",
    "BelowObjectPlane",
    "SurfaceToPlaneBridge",
    "Combined",
]
