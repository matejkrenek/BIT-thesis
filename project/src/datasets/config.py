from dataclasses import dataclass, field
import os


@dataclass
class DatasetDownloadConfig:
    repo_id: str = "ShapeNet/ShapeNetCore"
    token: str = os.getenv("HUGGING_FACE_TOKEN", "")
    local_dir: str = "data/raw/ShapeNetCore"


@dataclass
class DatasetConfig:
    name: str = "ShapeNetSynthetic"
    download: DatasetDownloadConfig = field(default_factory=DatasetDownloadConfig)
