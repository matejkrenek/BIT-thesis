import os
import os.path as osp
from typing import Callable, List, Optional, Union
from tqdm import tqdm

import torch
import numpy as np
import open3d as o3d
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

from dataset.downloader.huggingface import HuggingFaceDownloader
from logger import logger


class ShapeNetDataset(InMemoryDataset):

    category_ids = {
        "airplane": "02691156",
        "ashcan": "02747177",
        "bag": "02773838",
        "basket": "02801938",
        "bathtub": "02808440",
        "bed": "02818832",
        "bench": "02828884",
        "birdhouse": "02843684",
        "bookshelf": "02871439",
        "bottle": "02876657",
        "bowl": "02880940",
        "bus": "02924116",
        "cabinet": "02933112",
        "camera": "02942699",
        "can": "02946921",
        "cap": "02954340",
        "car": "02958343",
        "chair": "03001627",
        "clock": "03046257",
        "keyboard": "03085013",
        "dishwasher": "03207941",
        "display": "03211117",
        "earphone": "03261776",
        "faucet": "03325088",
        "file": "03337140",
        "guitar": "03467517",
        "helmet": "03513137",
        "jar": "03593526",
        "knife": "03624134",
        "lamp": "03636649",
        "laptop": "03642806",
        "loudspeaker": "03691459",
        "mailbox": "03710193",
        "microphone": "03759954",
        "microwave": "03761084",
        "motorcycle": "03790512",
        "mug": "03797390",
        "piano": "03928116",
        "pillow": "03938244",
        "pistol": "03948459",
        "pot": "03991062",
        "printer": "04004475",
        "remotecontrol": "04074963",
        "rifle": "04090263",
        "rocket": "04099429",
        "skateboard": "04225987",
        "sofa": "04256520",
        "stove": "04330267",
        "table": "04379243",
        "telephone": "04401088",
        "tower": "04460130",
        "train": "04468005",
        "vessel": "04530566",
        "washer": "04554684",
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
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logger.warning("HUGGINGFACE_TOKEN not set; download may fail.")

        downloader = HuggingFaceDownloader(
            local_dir=self.raw_dir,
            remote_dir="ShapeNet/ShapeNetCore",
            token=token,
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

            obj_ids = [
                d for d in os.listdir(cat_dir) if osp.isdir(osp.join(cat_dir, d))
            ]

            for obj_id in tqdm(obj_ids, desc=f"{category}", leave=False):
                obj_path = osp.join(cat_dir, obj_id, "models", "model_normalized.obj")
                if not osp.exists(obj_path):
                    continue

                try:
                    mesh = o3d.io.read_triangle_mesh(obj_path)

                    if not mesh.has_triangles():
                        continue

                    mesh.compute_vertex_normals()

                    vertices = np.asarray(mesh.vertices).astype(np.float32)
                    faces = np.asarray(mesh.triangles)

                    # Sample 8192 points (4096 is possible too)
                    pcd = mesh.sample_points_uniformly(number_of_points=8192)
                    pc_points = np.asarray(pcd.points).astype(np.float32)

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
                    out_file = osp.join(self.processed_dir, f"{category}_{obj_id}.pt")
                    torch.save(data, out_file)

                except Exception as e:
                    logger.error(f"Failed to process {obj_path}: {e}")
