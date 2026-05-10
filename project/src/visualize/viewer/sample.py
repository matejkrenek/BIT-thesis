"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: sample.py
Responsibility: Implements a viewer for visualizing dataset samples, including original, defected, and inferred point clouds.

This viewer is intended for interactive exploration of dataset samples and inference results using Polyscope.
"""

import polyscope as ps
import polyscope.imgui as psim
from .base import BaseViewer
from torch.utils.data import Dataset, DataLoader
from typing import Union, Callable, Optional


class SampleViewer(BaseViewer):
    """
    Viewer for visualizing samples from a dataset.

    Args:
        dataset (Union[Dataset, DataLoader]): The dataset or dataloader to visualize samples from.
        inference (Optional[Callable]): Optional function that takes a sample and returns a point cloud (for inference visualization).
    """

    def __init__(
        self,
        dataset: Union[Dataset, DataLoader],
        inference: Optional[Callable] = None,
    ):
        super().__init__()

        if not self.initialized:
            ps.init()
            # Store dataset and dataloader references
            self.dataset = dataset if isinstance(dataset, Dataset) else dataset.dataset
            self.dataloader = dataset if isinstance(dataset, DataLoader) else None
            self.inference = inference
            self.index = 0
            self.sample = self.dataset[self.index]
            self.initialized = True

    def gui_callback(self):
        """
        Draw the GUI elements for the viewer (navigation, info, toggles).
        """
        old_index = self.index

        # Navigation buttons and keyboard shortcuts
        if ps.imgui.Button("Previous") or psim.IsKeyPressed(psim.ImGuiKey_LeftArrow):
            self.prev()

        ps.imgui.SameLine()

        if ps.imgui.Button("Next") or psim.IsKeyPressed(psim.ImGuiKey_RightArrow):
            self.next()

        # Toggle visibility of defected/original clouds with arrow keys
        if psim.IsKeyPressed(psim.ImGuiKey_UpArrow):
            pc_defected = ps.get_point_cloud("defected")
            pc_defected.set_enabled(not pc_defected.is_enabled())

        if psim.IsKeyPressed(psim.ImGuiKey_DownArrow):
            pc_original = ps.get_point_cloud("original")
            pc_original.set_enabled(not pc_original.is_enabled())

        ps.imgui.Text(f"Sample {self.index + 1} / {len(self.dataset)}")

        # Display defect log only for tuple/list samples where the 3rd item is a dict.
        sample_log = None
        if isinstance(self.sample, (tuple, list)) and len(self.sample) > 2:
            maybe_log = self.sample[2]
            if isinstance(maybe_log, dict):
                sample_log = maybe_log

        if sample_log is not None:
            for defect, params in sample_log.items():
                ps.imgui.Separator()
                ps.imgui.TextColored((1.0, 1.0, 1.0, 1.0), f"Applied {defect}:")
                for key, value in params.items():
                    ps.imgui.BulletText(f"{key}: {value}")

        # Redraw only if sample has changed
        if old_index != self.index:
            self.draw()

    def next(self):
        """
        Go to the next sample in the dataset.
        """
        if self.index < len(self.dataset) - 1:
            self.index += 1
            self.sample = self.dataset[self.index]

    def prev(self):
        """
        Go to the previous sample in the dataset.
        """
        if self.index > 0:
            self.index -= 1
            self.sample = self.dataset[self.index]

    def draw(self):
        """
        Draw the current sample's point clouds (original, defected, and optionally inferred).
        """
        self.clear()

        # Register the original point cloud (usually ground truth)
        ps.register_point_cloud(
            "original",
            self.sample.original_pos,
            radius=0.00035,
            color=(0.0, 1.0, 0.0),
            point_render_mode="quad",
        )
        # Register the defected point cloud (with simulated defects)
        ps.register_point_cloud(
            "defected",
            self.sample.defected_pos,
            radius=0.00035,
            color=(1.0, 0.0, 0.0),
            point_render_mode="quad",
        )

        # Register inference result if inference function is provided
        if self.inference is not None:
            inferred_pc = self.inference(self.sample)
            ps.register_point_cloud(
                "inferred",
                inferred_pc,
                radius=0.00035,
                color=(0.0, 0.0, 1.0),
                point_render_mode="quad",
            )

    def show(self):
        """
        Show the viewer window and start the interactive Polyscope session.
        """
        self.draw()
        ps.set_user_callback(self.gui_callback)
        ps.set_ground_plane_mode("none")
        ps.show()

    def clear(self):
        """
        Clear all registered structures from the Polyscope window.
        """
        ps.remove_all_structures()
