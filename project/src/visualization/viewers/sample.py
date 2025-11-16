import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from .base import BaseViewer


class SampleViewer(BaseViewer):
    """Viewer for paginating through (clean, corrupted) point cloud pairs."""

    def __init__(self, pairs, shift=0.1, radius=0.003):
        super().__init__()
        if not self.initialized:
            ps.init()
            self.initialized = True
        self.pairs = pairs
        self.index = 0
        self.shift = shift
        self.radius = radius

    def next(self):
        self.cleanPCN.set_enabled(True)
        self.corruptedPCN.set_enabled(False)

    def prev(self):
        self.cleanPCN.set_enabled(False)
        self.corruptedPCN.set_enabled(True)

    def draw_pair(self, clean: np.ndarray, corrupted: np.ndarray):
        self.clear()

        clean_shifted = clean.copy()
        clean_shifted[:, 0] -= self.shift

        corrupted_shifted = corrupted.copy()
        corrupted_shifted[:, 0] += self.shift

        self.cleanPCN = ps.register_point_cloud("clean", clean_shifted)
        self.cleanPCN.set_radius(self.radius)
        self.corruptedPCN = ps.register_point_cloud("corrupted", corrupted_shifted)
        self.corruptedPCN.set_radius(self.radius)

        def callback():
            psim.Text("Use left/right arrow keys to navigate samples.")

            if psim.Button("Left") or psim.IsKeyPressed(psim.ImGuiKey_LeftArrow):
                self.prev()
            if psim.Button("Right") or psim.IsKeyPressed(psim.ImGuiKey_RightArrow):
                self.next()

        ps.set_user_callback(callback)

    def show(self):
        clean, corrupted = self.pairs[self.index]
        self.draw_pair(clean, corrupted)
        ps.show()

    def clear(self):
        ps.remove_all_structures()
