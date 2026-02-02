import os
import os.path as osp
from typing import Callable, List, Optional, Union
from tqdm import tqdm

import torch
import numpy as np
import open3d as o3d
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

from dataset.downloader import KaggleDownloader
from logger import logger

"""
ModelNet Dataset for 3D Point Cloud Classification.
Kaggle link: https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset
"""


class ModelNetDataset(InMemoryDataset):

    category_ids = {
        "airplane": "airplane",
        "bathtub": "bathtub",
        "bed": "bed",
        "bench": "bench",
        "bookshelf": "bookshelf",
        "bottle": "bottle",
        "bowl": "bowl",
        "car": "car",
        "chair": "chair",
        "cone": "cone",
        "cup": "cup",
        "curtain": "curtain",
        "desk": "desk",
        "door": "door",
        "dresser": "dresser",
        "flower_pot": "flower_pot",
        "glass_box": "glass_box",
        "guitar": "guitar",
        "keyboard": "keyboard",
        "lamp": "lamp",
        "laptop": "laptop",
        "mantel": "mantel",
        "monitor": "monitor",
        "night_stand": "night_stand",
        "person": "person",
        "piano": "piano",
        "plant": "plant",
        "radio": "radio",
        "range_hood": "range_hood",
        "sink": "sink",
        "sofa": "sofa",
        "stairs": "stairs",
        "stool": "stool",
        "table": "table",
        "tent": "tent",
        "toilet": "toilet",
        "tv_stand": "tv_stand",
        "vase": "vase",
        "wardrobe": "wardrobe",
        "xbox": "xbox",
    }

    def __init__(
        self,
        root: str,
        categories: Optional[Union[str, List[str]]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.root = root

        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]

        self.categories = categories
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super().__init__(root, log=False)

        if force_reload:
            logger.info("Force reload requested; clearing processed directory.")
            self._remove_processed()

        # If processed/ is empty → process automatically
        if len(self.processed_file_names) == 0:
            logger.info("Processed dataset not found. Processing dataset now...")
            self.process()

        # Build final file list
        self.files = sorted(os.listdir(self.processed_dir))

    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return list(self.category_ids.values())

    @property
    def processed_file_names(self) -> List[str]:
        if not osp.exists(self.processed_dir):
            return []
        return [f for f in os.listdir(self.processed_dir) if f.endswith(".pt")]

    def download(self):
        downloader = KaggleDownloader(
            local_dir=self.raw_dir,
            remote_dir="balraj98/modelnet40-princeton-3d-object-dataset",
        )
        downloader.download()

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = osp.join(self.processed_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)

        if self.transform:
            data = self.transform(data)
        return data

    def _remove_processed(self):
        if osp.exists(self.processed_dir):
            for f in os.listdir(self.processed_dir):
                os.remove(osp.join(self.processed_dir, f))

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        logger.info(f"Processing categories: {self.categories}")

        for category in tqdm(self.categories, desc="Categories"):
            cat_id = self.category_ids[category]
            cat_dir = osp.join(self.raw_dir, cat_id)

            if not osp.exists(cat_dir):
                logger.warning(f"Category {category} missing in raw/. Skipping.")
                continue

            for split in ["train", "test"]:
                split_dir = osp.join(cat_dir, split)
                if not osp.exists(split_dir):
                    continue

                off_files = [f for f in os.listdir(split_dir) if f.endswith(".off")]

                for off_file in tqdm(
                    off_files, desc=f"{category}/{split}", leave=False
                ):
                    off_path = osp.join(split_dir, off_file)

                    try:
                        mesh = o3d.io.read_triangle_mesh(off_path)

                        if not mesh.has_triangles():
                            continue

                        mesh.compute_vertex_normals()

                        vertices = np.asarray(mesh.vertices).astype(np.float32)
                        faces = np.asarray(mesh.triangles)

                        # Sample 8192 points (4096 is possible too)
                        pcd = mesh.sample_points_uniformly(number_of_points=8192)
                        pc_points = np.asarray(pcd.points, dtype=np.float32)

                        data = Data(
                            pos=torch.from_numpy(pc_points),
                            mesh_pos=torch.from_numpy(vertices),
                            face=torch.from_numpy(faces).t().long(),
                            category=category,
                        )

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        # Save each sample to disk → memory safe
                        out_file = osp.join(
                            self.processed_dir,
                            f"{category}_{split}_{off_file.replace('.off', '')}.pt",
                        )
                        torch.save(data, out_file)

                    except Exception as e:
                        logger.error(f"Failed to process {off_path}: {e}")
