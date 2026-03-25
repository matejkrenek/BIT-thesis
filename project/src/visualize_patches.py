import os
import torch
import polyscope as ps
from dotenv import load_dotenv
from dataset import ShapeNetDataset
from dataset.wrapper import (
    DenseWrapperDataset,
    NormalizeWrapperDataset,
    PatchWrapperDataset,
)

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "")
ROOT_DATA = DATA_FOLDER_PATH + "/data/ShapeNetV2"

# Build pipeline
base_dataset = ShapeNetDataset(root=ROOT_DATA)
dense_dataset = DenseWrapperDataset(
    dataset=base_dataset,
    root=DATA_FOLDER_PATH + "/data/ShapeNetV2_dense",
    num_points=100_000,
)
normalized_dataset = NormalizeWrapperDataset(dense_dataset)

# Create patch datasets
patch_dataset_single = PatchWrapperDataset(
    normalized_dataset,
    patch_size=1024,
    num_patches=1,
    sampling_strategy="random",
    seed=42,
)

patch_dataset_multi = PatchWrapperDataset(
    normalized_dataset,
    patch_size=1024,
    num_patches=4,
    sampling_strategy="fps",
    seed=42,
)

# Initialize polyscope
ps.init()

# Get sample index (can be changed)
idx = 0

print("\n" + "=" * 60)
print("PATCH VISUALIZATION")
print("=" * 60)

# Visualize full normalized cloud
full_sample = normalized_dataset[idx]
ps.register_point_cloud(
    f"Full Cloud (idx={idx})",
    full_sample.pos.numpy(),
    color=(0.8, 0.8, 0.8),
    enabled=True,
)
print(f"Full cloud:  {full_sample.pos.shape}")

# Visualize single patch
print("\n--- Single Random Patch ---")
single_patch = patch_dataset_single[idx]
ps.register_point_cloud(
    "Single Patch (random)",
    single_patch.pos.numpy(),
    color=(0.2, 0.6, 0.9),  # Blue
    enabled=True,
)
print(f"Patch shape: {single_patch.pos.shape}")

# Visualize multiple patches
print("\n--- Multiple FPS Patches ---")
multi_patches = patch_dataset_multi[idx]
patches_np = multi_patches.pos.numpy()  # [4, 1024, 3]

colors = [
    (0.9, 0.2, 0.2),  # Red
    (0.2, 0.9, 0.2),  # Green
    (0.9, 0.9, 0.2),  # Yellow
    (0.9, 0.2, 0.9),  # Magenta
]

for i, patch in enumerate(patches_np):
    ps.register_point_cloud(
        f"Patch {i+1} (fps)",
        patch,
        color=colors[i],
        enabled=True,
    )
    print(f"Patch {i+1}: {patch.shape}")

# Print statistics
print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)
print(f"Full cloud points:    {full_sample.pos.shape[0]}")
print(f"Single patch points:  {single_patch.pos.shape[0]}")
print(f"Multi patches:        {multi_patches.num_patches}")
print(f"Points per patch:     {multi_patches.patch_size}")
print(
    f"Coverage:             {(multi_patches.num_patches * multi_patches.patch_size / full_sample.pos.shape[0] * 100):.1f}%"
)
print("=" * 60 + "\n")

# Launch viewer
print("Opening polyscope viewer...")
print("Controls:")
print("  - Scroll to zoom")
print("  - Right-click drag to rotate")
print("  - Left-click to select")
print("  - Spacebar to show/hide point cloud list")
ps.show()
