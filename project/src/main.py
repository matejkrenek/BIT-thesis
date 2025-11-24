from dataset.shapenet import ShapeNetDataset
from dataset.augmented import AugmentedDataset
from dataset.defects import Noise, Rotate, Scale, Translate
from visualize.viewers import SampleViewer
import polyscope as ps

dataset = AugmentedDataset(
    dataset=ShapeNetDataset(root="data/ShapeNet", split="train", categories=["Pistol"]),
    defects=[
        Noise(sigma=0.01),
        Rotate(z=45),
        Scale(factor=1.2),
        Translate(x=0.1, y=0.0, z=0.0),
    ],
)


def render_callback(sample):
    ps.register_point_cloud(
        "original",
        sample.original,
        radius=0.005,
        color=(0.0, 1.0, 0.0),
    )
    ps.register_point_cloud(
        "defected",
        sample.defected,
        radius=0.005,
        color=(1.0, 0.0, 0.0),
    )


def text_callback(sample):
    for defect, params in sample.log.items():
        ps.imgui.Separator()
        ps.imgui.TextColored((1.0, 1.0, 1.0, 1.0), f"Applied {defect}:")
        for key, value in params.items():
            ps.imgui.BulletText(f"{key}: {value}")


viewer = SampleViewer(
    dataset=dataset,
    render_callback=render_callback,
    text_callback=text_callback,
)
viewer.show()
# # Initialize polyscope
# ps.init()

# # Convert loader to list for easier navigation
# samples = list(loader)
# current_idx = [0]  # Use list to allow modification in nested function


# def show_sample(idx):
#     """Display the sample at the given index."""
#     ps.remove_all_structures()

#     if 0 <= idx < len(samples):
#         data = samples[idx]
#         points = data["original"][0]
#         ps.register_point_cloud(
#             f"sample_{idx}", points, radius=0.01, color=(1.0, 0.0, 0.0)
#         )
#         print(f"Showing sample {idx + 1}/{len(samples)}")


# def callback():
#     """Callback to handle navigation between samples."""
#     changed = False

#     if ps.imgui.Button("Previous"):
#         if current_idx[0] > 0:
#             current_idx[0] -= 1
#             changed = True

#     ps.imgui.SameLine()

#     if ps.imgui.Button("Next"):
#         if current_idx[0] < len(samples) - 1:
#             current_idx[0] += 1
#             changed = True

#     ps.imgui.Text(f"Sample {current_idx[0] + 1} / {len(samples)}")

#     if changed:
#         show_sample(current_idx[0])


# # Show first sample
# show_sample(0)

# # Set callback and show
# ps.set_user_callback(callback)
# ps.show()
