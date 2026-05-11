"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: types.py
Responsibility: Type aliases for reduction strategies in point cloud metrics.
"""

from typing import Literal

Reduction = Literal["mean", "sum", "none"]

__all__ = ["Reduction"]
