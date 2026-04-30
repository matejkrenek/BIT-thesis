"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: __init__.py
Responsibility: Exports synthetic defect classes for point cloud corruption and augmentation.
"""

from .base import Defect
from .noise import Noise
from .rotate import Rotate
from .scale import Scale
from .translate import Translate
from .local_dropout import LocalDropout
from .floating_cluster import FloatingCluster
from .outlier_points import OutlierPoints
from .large_missing_region import LargeMissingRegion
from .bridging_artifact import BridgingArtifact
from .surface_bridging_artifact import SurfaceBridgingArtifact
from .hair_like_noise import HairLikeNoise
from .surface_flattening import SurfaceFlattening
from .surface_bulging import SurfaceBulging
from .anisotropic_stretch_noise import AnisotropicStretchNoise
from .below_object_plane import BelowObjectPlane
from .surface_to_plane_bridge import SurfaceToPlaneBridge
from .combined import Combined

__all__ = [
    "Defect",
    "Noise",
    "Rotate",
    "Scale",
    "Translate",
    "LocalDropout",
    "FloatingCluster",
    "OutlierPoints",
    "LargeMissingRegion",
    "BridgingArtifact",
    "SurfaceBridgingArtifact",
    "HairLikeNoise",
    "SurfaceFlattening",
    "SurfaceBulging",
    "AnisotropicStretchNoise",
    "BelowObjectPlane",
    "SurfaceToPlaneBridge",
    "Combined",
]
