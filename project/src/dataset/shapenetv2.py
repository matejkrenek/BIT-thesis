import os
import os.path as osp
import shutil
import glob
from typing import Callable, List, Optional, Union

import torch
import numpy as np
import open3d as o3d
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_obj
import torch_geometric.transforms as T
from dotenv import load_dotenv

from dataset.downloader.huggingface import HuggingFaceDownloader
from logger import logger


class ShapeNetV2Dataset(InMemoryDataset):

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
        include_normals: bool = True,
        split: str = "trainval",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:

        return list(self.category_ids.values())

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logger.warning(
                "HUGGINGFACE_TOKEN not found in environment variables. Download might fail if dataset is private."
            )

        downloader = HuggingFaceDownloader(
            local_dir=self.raw_dir, remote_dir="ShapeNet/ShapeNetCore", token=token
        )

        downloader.download()

    def process(self) -> None:
        data_list = []

        for category in self.categories:
            cat_id = self.category_ids[category]
            cat_dir = osp.join(self.raw_dir, cat_id)

            if not osp.exists(cat_dir):
                logger.warning(f"Category directory {cat_dir} not found. Skipping.")
                continue

            # ShapeNetCore v2 structure: category_id / object_id / models / model_normalized.obj
            # We look for all subdirectories
            obj_ids = [
                d for d in os.listdir(cat_dir) if osp.isdir(osp.join(cat_dir, d))
            ]

            for obj_id in obj_ids:
                obj_path = osp.join(cat_dir, obj_id, "models", "model_normalized.obj")
                if not osp.exists(obj_path):
                    continue

                try:
                    # Use Open3D to read the mesh as it is more robust to different OBJ formats
                    mesh = o3d.io.read_triangle_mesh(obj_path)

                    # Convert to PyG Data
                    pos = torch.from_numpy(np.asarray(mesh.vertices)).float()
                    face = torch.from_numpy(np.asarray(mesh.triangles)).t().long()

                    data = Data(pos=pos, face=face)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                except Exception as e:
                    logger.error(f"Failed to process {obj_path}: {e}")

        if not data_list:
            logger.warning("No data processed!")
            return

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self)}, " f"categories={self.categories})"
        )
