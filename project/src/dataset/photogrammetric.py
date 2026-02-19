from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from itertools import combinations
import numpy as np
import torch
from typing import Union, List
import os
import open3d as o3d
from pathlib import Path
from mini_dust3r.api import (
    OptimizedResult,
    inferece_dust3r,
    log_optimized_result,
)
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.utils.image import load_images
import cv2
import trimesh
from notifications import DiscordNotifier

class PhotogrammetricDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        frames_per_sample: Union[int, List[int]] = 15,
        frames_strategy: str = "uniform",
        force_reload: bool = False,

    ):
        self.base = dataset
        self.frames_per_sample = frames_per_sample
        self.frames_strategy = frames_strategy
        self.force_reload = force_reload
        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/1469079470815318237/cb9MA1kXMvg-wM5t_GYMUJNe7-eUhT2aRCCoaB-Bq022XvoCsANa5yz1ySn2VpP2TuN_",
            project_name="BIT Thesis Project",
            project_url="https://github.com/matejkrenek/BIT-thesis",
            avatar_name="Photogrammetric Dataset Bot",
        )

    def __len__(self):
        return len(self.base) * (1 if isinstance(self.frames_per_sample, int) else len(self.frames_per_sample))

    @staticmethod
    def extract_frames(
        image_dir_or_list: Union[Path, str, List[Path], List[str]],
        num_frames: int,
        strategy: str = "uniform",
    ) -> List[Path]:
        """
        Extract a subset of frames from video frames for DUSt3R reconstruction.
        
        Args:
            image_dir_or_list: Either a directory containing video frames or a list of image paths
            num_frames: Number of frames to extract (e.g., 20 from 100)
            strategy: Frame selection strategy:
                - "uniform": Evenly distribute frames across the sequence
                - "keyframe": Select frames with maximum spacing for better coverage
                - "random": Random selection (with seed for reproducibility)
            
        Returns:
            List of selected frame paths
        """
        # Handle input - convert to list of Path objects
        if isinstance(image_dir_or_list, (str, Path)):
            image_dir = Path(image_dir_or_list)
            if not image_dir.exists():
                raise ValueError(f"Directory does not exist: {image_dir}")
            
            # Get all image files, sorted by name
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            all_frames = []
            for ext in image_extensions:
                all_frames.extend(image_dir.glob(f'*{ext}'))
                all_frames.extend(image_dir.glob(f'*{ext.upper()}'))
            
            all_frames = sorted(all_frames, key=lambda x: x.name)
            
        else:
            # Handle list of image paths
            all_frames = [Path(img) for img in image_dir_or_list]
            all_frames = sorted(all_frames, key=lambda x: x.name)
        
        if len(all_frames) == 0:
            raise ValueError("No image files found")
        
        if num_frames >= len(all_frames):
            print(f"Warning: Requested {num_frames} frames but only {len(all_frames)} available. Using all frames.")
            selected_frames = all_frames
        else:
            # Select frames based on strategy
            if strategy == "uniform":
                # Evenly distribute frames across the sequence
                indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
                selected_frames = [all_frames[i] for i in indices]
                
            elif strategy == "keyframe":
                # Maximum spacing strategy - good for reconstruction
                if num_frames == 1:
                    selected_frames = [all_frames[len(all_frames) // 2]]
                else:
                    # Always include first and last frame for temporal coverage
                    indices = [0, len(all_frames) - 1]
                    
                    # Add intermediate frames with maximum spacing
                    remaining_frames = num_frames - 2
                    if remaining_frames > 0:
                        intermediate_indices = np.linspace(1, len(all_frames) - 2, remaining_frames, dtype=int)
                        indices.extend(intermediate_indices)
                    
                    indices = sorted(set(indices))  # Remove duplicates and sort
                    selected_frames = [all_frames[i] for i in indices[:num_frames]]
                    
            elif strategy == "random":
                # Random selection with seed for reproducibility
                np.random.seed(42)
                indices = np.random.choice(len(all_frames), size=num_frames, replace=False)
                indices = sorted(indices)
                selected_frames = [all_frames[i] for i in indices]
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'uniform', 'keyframe', or 'random'")
        
        print(f"Selected {len(selected_frames)} frames from {len(all_frames)} total frames using '{strategy}' strategy")
        
        return [str(frame) for frame in selected_frames]

    @staticmethod
    def apply_masks_to_pointcloud(
        optimized_results,
        mask_paths,
        min_votes: int = 1,
        depth_tau: float | None = None,
        conf_thresh: float | None = None,
    ):
        """
        Filters DUSt3R point cloud using:
        - external SAM object masks
        - multi-view voting
        - depth consistency
        - optional confidence filtering

        Args:
            optimized_results: OptimizedResult from mini-dust3r
            mask_paths: list of SAM mask paths (same order as input images)
            min_votes: minimum number of views required
            depth_tau: relative depth tolerance (e.g. 0.05 = 5%)
            conf_thresh: confidence threshold (None to disable)

        Returns:
            trimesh.PointCloud (masked)
        """

        # ------------------------------------------------------------
        # Load masks
        # ------------------------------------------------------------
        sam_masks = []
        for i, path in enumerate(mask_paths):
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(path)

            sam_masks.append(m > 0)

        depth_maps = optimized_results.depth_hw_list
        conf_maps = optimized_results.conf_hw_list if conf_thresh is not None else None

        # ------------------------------------------------------------
        # Prepare geometry
        # ------------------------------------------------------------
        points = np.asarray(optimized_results.point_cloud.vertices)  # (N, 3)
        colors = optimized_results.point_cloud.colors
        N = len(points)

        points_h = np.hstack([points, np.ones((N, 1))])  # (N, 4)
        votes = np.zeros(N, dtype=np.int32)

        K_all = optimized_results.K_b33
        T_all = optimized_results.world_T_cam_b44

        # ------------------------------------------------------------
        # Multi-view voting (vectorized per view)
        # ------------------------------------------------------------
        for i, (K, T, sam_mask) in enumerate(zip(K_all, T_all, sam_masks)):

            # world -> camera
            pts_cam = (T @ points_h.T).T[:, :3]

            # points in front of camera
            valid = pts_cam[:, 2] > 0
            if not np.any(valid):
                continue

            pts_cam = pts_cam[valid]
            idx_valid = np.where(valid)[0]

            # project to image plane
            proj = (K @ pts_cam.T).T
            uv = proj[:, :2] / proj[:, 2:3]

            u = uv[:, 0].astype(int)
            v = uv[:, 1].astype(int)

            H, W = sam_mask.shape
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not np.any(inside):
                continue

            idx = idx_valid[inside]
            u = u[inside]
            v = v[inside]

            # --------------------------------------------------------
            # Conditions
            # --------------------------------------------------------
            # 1) SAM object mask
            cond = sam_mask[v, u]

            # 2) Depth consistency
            if depth_tau is not None:
                z_proj = pts_cam[inside, 2]
                z_ref = depth_maps[i][v, u]
                cond &= (z_proj <= z_ref * (1.0 + depth_tau))

            # 3) Confidence (optional)
            if conf_maps is not None:
                cond &= conf_maps[i][v, u] > conf_thresh

            votes[idx[cond]] += 1

        # ------------------------------------------------------------
        # Final filtering
        # ------------------------------------------------------------
        keep = votes >= min_votes

        return trimesh.PointCloud(
            vertices=points[keep],
            colors=colors[keep] if colors is not None else None,
        )

   


    def __getitem__(self, idx):
        try:
            base_idx = idx // (1 if isinstance(self.frames_per_sample, int) else len(self.frames_per_sample))
            variant_id = idx % (1 if isinstance(self.frames_per_sample, int) else len(self.frames_per_sample))

            data = self.base[base_idx]

            if not isinstance(data, Data) or not hasattr(data, "images") or not hasattr(data, "masks"):
                return None  # vadný vzorek

            frames_per_sample = self.frames_per_sample if isinstance(self.frames_per_sample, int) else self.frames_per_sample[variant_id]
            pointcloud_unmasked_path = data.sample_dir + f"/pointcloud_{frames_per_sample}_{self.frames_strategy}_unmasked.ply"
            pointcloud_masked_path = data.sample_dir + f"/pointcloud_{frames_per_sample}_{self.frames_strategy}_masked.ply"

            if (os.path.exists(pointcloud_masked_path) or os.path.exists(pointcloud_unmasked_path)) and not self.force_reload:
                pointcloud_unmasked = trimesh.load(pointcloud_unmasked_path)
                pointcloud_masked = trimesh.load(pointcloud_masked_path)
                
                return (
                    pointcloud_unmasked,
                    pointcloud_masked,
                )

            self.notifier.send_custom_notification(
                title=f"Processing sample {idx}",
                description=f"Base index: {base_idx}, Variant ID: {variant_id}",
                color="3498db",
                fields=[
                    {"name": "Frames per sample", "value": str(frames_per_sample), "inline": True},
                    {"name": "Strategy", "value": self.frames_strategy, "inline": True},
                    {"name": "Unmasked path", "value": pointcloud_unmasked_path, "inline": False},
                    {"name": "Masked path", "value": pointcloud_masked_path, "inline": False},
                ]
            )
            
            images = data.images
            masks = data.masks

            selected_frame_paths = self.extract_frames(
                image_dir_or_list=images,
                num_frames=frames_per_sample,
                strategy=self.frames_strategy,
            )
            selected_mask_paths = [masks[i] for i, img in enumerate(data.images) if img in selected_frame_paths]

            optimized_results: OptimizedResult = inferece_dust3r(
                image_dir_or_list=selected_frame_paths,
                model=self.model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=2,
                niter=200
            )

            pointcloud_unmasked = optimized_results.point_cloud
            pointcloud_unmasked.export(pointcloud_unmasked_path)

            # # Apply masks to point cloud
            pointcloud_masked = self.apply_masks_to_pointcloud(
                optimized_results,
                selected_mask_paths,
                min_votes=1,
                depth_tau=None,
                conf_thresh=None,
            )

            pointcloud_masked.export(pointcloud_masked_path)

            self.notifier.send_custom_notification(
                title=f"Finished processing sample {idx}",
                color="2ecc71",
                description=f"Saved unmasked and masked point clouds.",
                fields=[
                    {"name": "Unmasked path", "value": pointcloud_unmasked_path, "inline": False},
                    {"name": "Masked path", "value": pointcloud_masked_path, "inline": False},
                ]
            )

            return (
                pointcloud_unmasked,
                pointcloud_masked,
            )
        except Exception as e:
            self.notifier.send_custom_notification(
                title=f"Error processing sample {idx}",
                color="e74c3c",
                description="An error occurred during processing.",
                fields=[
                    {"name": "Error Message", "value": f"```{str(e)}```", "inline": False},
                ]
            )
            raise e



