import os
import torch
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

# Test patch wrapper with different strategies
print("Testing PatchWrapperDataset:\n")

# Test 1: Single random patch
print("=" * 60)
print("Test 1: Single random patch (patch_size=512)")
print("=" * 60)
patch_dataset_random = PatchWrapperDataset(
    normalized_dataset,
    patch_size=512,
    num_patches=1,
    sampling_strategy="random",
    seed=42,
)

sample = patch_dataset_random[0]
print(f"Output shape:     {sample.pos.shape}")
print(f"Expected:         torch.Size([512, 3])")
print(f"Category:         {sample.category}")
print(f"Patch size:       {sample.patch_size}")
print(f"Num patches:      {sample.num_patches}")
print(f"Has NaN:          {torch.isnan(sample.pos).any().item()}")
print(f"Has Inf:          {torch.isinf(sample.pos).any().item()}")
print()

# Test 2: Multiple random patches
print("=" * 60)
print("Test 2: Multiple random patches (num_patches=4)")
print("=" * 60)
patch_dataset_multi = PatchWrapperDataset(
    normalized_dataset,
    patch_size=512,
    num_patches=4,
    sampling_strategy="random",
    seed=42,
)

sample = patch_dataset_multi[0]
print(f"Output shape:     {sample.pos.shape}")
print(f"Expected:         torch.Size([4, 512, 3])")
print(f"Category:         {sample.category}")
print(f"Patch size:       {sample.patch_size}")
print(f"Num patches:      {sample.num_patches}")
print(f"Has NaN:          {torch.isnan(sample.pos).any().item()}")
print(f"Has Inf:          {torch.isinf(sample.pos).any().item()}")
print()

# Test 3: FPS-based patches
print("=" * 60)
print("Test 3: FPS-based patches (sampling_strategy='fps')")
print("=" * 60)
patch_dataset_fps = PatchWrapperDataset(
    normalized_dataset,
    patch_size=512,
    num_patches=1,
    sampling_strategy="fps",
    seed=42,
)

sample = patch_dataset_fps[0]
print(f"Output shape:     {sample.pos.shape}")
print(f"Expected:         torch.Size([512, 3])")
print(f"Category:         {sample.category}")
print(f"Has NaN:          {torch.isnan(sample.pos).any().item()}")
print(f"Has Inf:          {torch.isinf(sample.pos).any().item()}")
print()

# Test 4: Reproducibility (same seed = same patch)
print("=" * 60)
print("Test 4: Reproducibility (same idx, same patch)")
print("=" * 60)
sample1 = patch_dataset_random[0]
sample2 = patch_dataset_random[0]
same = torch.allclose(sample1.pos, sample2.pos)
print(f"Same patch (same idx):  {same}")

sample3 = patch_dataset_random[1]
different = not torch.allclose(sample1.pos, sample3.pos)
print(f"Different patch (different idx): {different}")
print()

print("✓ All tests passed!")
