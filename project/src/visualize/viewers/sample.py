import polyscope as ps
import polyscope.imgui as psim
from .base import BaseViewer
from torch.utils.data import Dataset


class SampleViewer(BaseViewer):
    """Viewer for paginating through (clean, corrupted) point cloud pairs."""

    def __init__(
        self,
        dataset: Dataset,
        render_callback: callable = None,
        text_callback: callable = None,
    ):
        self.render_callback = render_callback
        self.text_callback = text_callback
        super().__init__()

        if not self.initialized:
            ps.init()
            self.dataset = dataset
            self.index = 0
            self.sample = dataset[self.index]
            self.initialized = True

    def gui_callback(self):
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

        if self.text_callback is not None:
            self.text_callback(self.sample)

        if old_index != self.index:
            self.draw()

    def next(self):
        if self.index < len(self.dataset) - 1:
            self.index += 1
            self.sample = self.dataset[self.index]

    def prev(self):
        if self.index > 0:
            self.index -= 1
            self.sample = self.dataset[self.index]

    def draw(self):
        self.clear()
        if self.render_callback is not None:
            self.render_callback(self.sample)

    def show(self):
        self.draw()
        ps.set_user_callback(self.gui_callback)
        ps.set_ground_plane_mode("none")

        ps.show()

    def clear(self):
        ps.remove_all_structures()
