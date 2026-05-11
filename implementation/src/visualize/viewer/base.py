"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: base.py
Responsibility: Defines the abstract base class for all 3D point cloud viewers in the visualize.viewer module.

This file is intended to be extended by concrete viewer implementations (e.g., for dataset samples, inference, etc.).
"""

from abc import ABC, abstractmethod


class BaseViewer(ABC):
    """
    Abstract base class for all viewers.

    Every viewer must implement the show() and clear() methods.
    """

    initialized = False

    @abstractmethod
    def show(self, *args, **kwargs):
        """
        Launch the main visualization window or loop.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear all visualized structures (e.g., remove all point clouds from the window).
        """
        pass
