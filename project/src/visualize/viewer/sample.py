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
        inference (Optional[Callable]): Optional function that takes a sample and returns a point cloud.
    """

    def __init__(
        self,
        dataset: Union[Dataset, DataLoader],
        inference: Optional[Callable] = None,
    ):
        super().__init__()

        if not self.initialized:
            ps.init()
            self.dataset = dataset if isinstance(dataset, Dataset) else dataset.dataset
            self.dataloader = dataset if isinstance(dataset, DataLoader) else None
            self.inference = inference
            self.index = 0
            self.sample = self.dataset[self.index]
            self.initialized = True

    def gui_callback(self):
        """Draw the GUI elements for the viewer."""
        old_index = self.index

        if ps.imgui.Button("Previous") or psim.IsKeyPressed(psim.ImGuiKey_LeftArrow):
            self.prev()

        ps.imgui.SameLine()

        if ps.imgui.Button("Next") or psim.IsKeyPressed(psim.ImGuiKey_RightArrow):
            self.next()

        if psim.IsKeyPressed(psim.ImGuiKey_UpArrow):
            pc_defected = ps.get_point_cloud("defected")
            pc_defected.set_enabled(not pc_defected.is_enabled())

        if psim.IsKeyPressed(psim.ImGuiKey_DownArrow):
            pc_original = ps.get_point_cloud("original")
            pc_original.set_enabled(not pc_original.is_enabled())

        ps.imgui.Text(f"Sample {self.index + 1} / {len(self.dataset)}")

        # Display defect log if available
        if len(self.sample) > 2 and isinstance(self.sample[2], dict):
            for defect, params in self.sample[2].items():
                ps.imgui.Separator()
                ps.imgui.TextColored((1.0, 1.0, 1.0, 1.0), f"Applied {defect}:")
                for key, value in params.items():
                    ps.imgui.BulletText(f"{key}: {value}")

        # Redraw only if sample has changed
        if old_index != self.index:
            self.draw()

    def next(self):
        """Go to the next sample in the dataset."""
        if self.index < len(self.dataset) - 1:
            self.index += 1
            self.sample = self.dataset[self.index]

    def prev(self):
        """Go to the previous sample in the dataset."""
        if self.index > 0:
            self.index -= 1
            self.sample = self.dataset[self.index]

    def draw(self):
        """Draw the current sample point clouds."""
        self.clear()

        ps.register_point_cloud(
            "original",
            self.sample.original_pos,
            radius=0.00035,
            color=(0.0, 1.0, 0.0),
            point_render_mode="quad",
        )
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
        """Show the viewer window."""
        self.draw()
        ps.set_user_callback(self.gui_callback)
        ps.set_ground_plane_mode("none")

        ps.show()

    def clear(self):
        """Clear all registered structures."""
        ps.remove_all_structures()
