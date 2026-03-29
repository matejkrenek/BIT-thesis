import os
import torch
import polyscope as ps
from dotenv import load_dotenv
from dataset import ShapeNetDataset
from dataset.wrapper import (
    DenseWrapperDataset,
    NormalizeWrapperDataset,
    PatchWrapperDataset,
    AugmentWrapperDataset,
)
from dataset.defect import LargeMissingRegion, Rotate, Noise, LocalDropout, Combined

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR", "")
ROOT_DATA = ROOT_DIR + "/data/ShapeNetV2"

# Build pipeline
base_dataset = ShapeNetDataset(root=ROOT_DATA)
dense_dataset = DenseWrapperDataset(
    dataset=base_dataset,
    root=ROOT_DIR + "/data/ShapeNetV2_dense",
    num_points=100_000,
)
normalized_dataset = NormalizeWrapperDataset(dense_dataset)
defects = [
    Combined(
        [
            LargeMissingRegion(removal_fraction=0.3),
            LocalDropout(
                radius=0.1,
                regions=5,
                dropout_rate=0.5,
            ),
        ]
    )
    for _ in range(5)
]


augmented_base_dataset = AugmentWrapperDataset(normalized_dataset, defects)

patch_dataset_multi = PatchWrapperDataset(
    dataset=augmented_base_dataset,
    patch_size=8192 * 4,
    num_patches=8,
    overlap_ratio=0,
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

# Visualize multiple patches
print("\n--- Multiple FPS+KNN Patches ---")
multi_patches = patch_dataset_multi[idx]
patches_defected_np = multi_patches.defected_pos.numpy()  # [num_patches, patch_size, 3]
patches_original_np = multi_patches.original_pos.numpy()  # [num_patches, patch_size, 3]

for i, patch in enumerate(patches_defected_np):
    ps.register_point_cloud(
        f"Patch {i+1} (fps+knn)",
        patch,
        enabled=False,
    )
    print(f"Patch {i+1}: {patch.shape}")

for i, patch in enumerate(patches_original_np):
    ps.register_point_cloud(
        f"Patch {i+1} (fps+knn2)",
        patch,
        enabled=False,
    )
    print(f"Patch {i+1}: {patch.shape}")

# Print statistics
print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)
print(f"Full cloud points:    {full_sample.pos.shape[0]}")
print(f"Multi patches:        {multi_patches.num_patches}")
print(f"Points per patch:     {multi_patches.patch_size}")
print(f"Overlap ratio:        {multi_patches.overlap_ratio:.2f}")
print(f"Coverage ratio:       {multi_patches.coverage_ratio * 100:.2f}%")
print(
    f"Sampling ratio:       {(multi_patches.num_patches * multi_patches.patch_size / full_sample.pos.shape[0] * 100):.1f}%"
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
