from pathlib import Path
from argparse import ArgumentParser
import torch
from typing import Union, List
import os
import shutil

from mini_dust3r.api import (
    OptimizedResult,
    inferece_dust3r,
    log_optimized_result,
)
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.utils.image import load_images
import numpy as np


def extract_from_trimesh_pc(pc):
    points = np.asarray(pc.vertices, dtype=np.float32)

    colors = np.asarray(pc.colors)
    if colors.shape[1] == 4:  # RGBA → RGB
        colors = colors[:, :3]

    # normalize for Polyscope
    if colors.max() > 1.0:
        colors = colors.astype(np.float32) / 255.0

    return points, colors


def extract_video_frames(
    image_dir_or_list: Union[Path, str, List[Path], List[str]],
    num_frames: int,
    output_dir: Path = None,
    strategy: str = "uniform",
    copy_files: bool = False
) -> List[Path]:
    """
    Extract a subset of frames from video frames for DUSt3R reconstruction.
    
    Args:
        image_dir_or_list: Either a directory containing video frames or a list of image paths
        num_frames: Number of frames to extract (e.g., 20 from 100)
        output_dir: Optional output directory to copy selected frames to
        strategy: Frame selection strategy:
            - "uniform": Evenly distribute frames across the sequence
            - "keyframe": Select frames with maximum spacing for better coverage
            - "random": Random selection (with seed for reproducibility)
        copy_files: Whether to copy files to output_dir (if provided) or just return paths
        
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
    
    # Optionally copy files to output directory
    if output_dir and copy_files:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        copied_frames = []
        for i, frame_path in enumerate(selected_frames):
            # Create sequential naming: frame_001.jpg, frame_002.jpg, etc.
            new_name = f"frame_{i+1:03d}{frame_path.suffix}"
            dest_path = output_dir / new_name
            shutil.copy2(frame_path, dest_path)
            copied_frames.append(dest_path)
        
        print(f"Copied selected frames to: {output_dir}")
        return copied_frames
    
    return [str(frame) for frame in selected_frames]


def main(image_dir: Path, num_frames: int = None, frame_strategy: str = "uniform", output_ply: Path = Path("reconstruction.ply")):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    # If num_frames is specified, extract subset of frames
    if num_frames is not None:
        print(f"Extracting {num_frames} frames from video sequence...")
        selected_frames = extract_video_frames(
            image_dir_or_list=image_dir,
            num_frames=num_frames,
            strategy=frame_strategy
        )
        
        # Use selected frames for reconstruction
        image_input = selected_frames
    else:
        # Use all frames in directory
        image_input = str(image_dir)

    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=image_input,
        model=model,
        device=device,
        batch_size=1,
    )

    optimized_results.point_cloud.export(output_ply)

    print(optimized_results)


if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r frame extraction and reconstruction script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="Number of frames to extract from the video sequence (e.g., 20 from 100)",
        default=None,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["uniform", "keyframe", "random"],
        default="uniform",
        help="Frame selection strategy: 'uniform' (evenly spaced), 'keyframe' (max coverage), or 'random'",
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        help="Output path for the reconstructed point cloud PLY file",
        default=Path("reconstruction.ply"),
    )
    args = parser.parse_args()

    main(args.image_dir, args.num_frames, args.strategy, args.output_ply)
