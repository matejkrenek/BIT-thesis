from pathlib import Path
import open3d as o3d
from logger import logger


class MeshLoader:
    def __init__(self, synset_dir: str):
        self.synset_dir = Path(synset_dir)

    def load_meshes(self):
        for model_dir in self.synset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            obj_path = model_dir / "models/model_normalized.obj"
            if not obj_path.exists():
                continue

            mesh = o3d.io.read_triangle_mesh(str(obj_path))
            if mesh.is_empty():
                logger.warning(f"Mesh empty: {obj_path}")
                continue

            yield mesh, model_dir.name
