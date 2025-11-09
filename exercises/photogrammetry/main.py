import pycolmap
from pathlib import Path
import os

# ----------------------------------------------------------------------
# 1️⃣ Paths setup
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
image_dir = BASE_DIR / "data" / "images"  # input images
output_path = BASE_DIR / "data" / "reconstruction"  # output root
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

# Create directories if missing
output_path.mkdir(parents=True, exist_ok=True)
mvs_path.mkdir(parents=True, exist_ok=True)

print("[INFO] Starting photogrammetry pipeline with COLMAP backend")

# ----------------------------------------------------------------------
# 2️⃣ Feature extraction + matching
# ----------------------------------------------------------------------
print("[STEP 1] Extracting SIFT features...")
pycolmap.extract_features(
    database_path,
    image_dir,
    sift_options=pycolmap.SiftExtractionOptions(num_threads=2),
)

print("[STEP 2] Matching features...")
pycolmap.match_exhaustive(database_path)

# ----------------------------------------------------------------------
# 3️⃣ Incremental mapping (Sparse reconstruction)
# ----------------------------------------------------------------------
print("[STEP 3] Running incremental mapping (Structure-from-Motion)...")
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
print(f"[INFO] Sparse model written to: {output_path}")

# ----------------------------------------------------------------------
# 4️⃣ Dense reconstruction (MVS)
# ----------------------------------------------------------------------
print("[STEP 4] Undistorting images...")
pycolmap.undistort_images(mvs_path, output_path, image_dir)

print("[STEP 5] PatchMatch stereo (depth maps)...")
try:
    pycolmap.patch_match_stereo(mvs_path)  # requires CUDA build
except Exception as e:
    print(f"[WARN] PatchMatch stereo skipped (likely missing CUDA): {e}")

print("[STEP 6] Stereo fusion (combine depth maps → dense cloud)...")
dense_ply = mvs_path / "dense.ply"
pycolmap.stereo_fusion(dense_ply, mvs_path)

print(f"[✅ DONE] Dense point cloud saved to: {dense_ply}")
print("[TIP] You can now visualize it with Open3D or run your QA analyzer.")
