import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path
import os

# Load image
image = cv2.imread(Path("./data/images/scissors/1769628755355.jpg"))

if image is None:
    raise FileNotFoundError("Image not found or cannot be read")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint=Path("./checkpoints/sam_vit_h_4b8939.pth"))
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

# Automatic mask generator
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
)

masks = mask_generator.generate(image)

print(f"Generated {len(masks)} masks")

# os.makedirs("masks", exist_ok=True)

# for i, m in enumerate(masks):
#     mask = m["segmentation"].astype(np.uint8) * 255
#     cv2.imwrite(f"masks/mask_{i:02d}.png", mask)
