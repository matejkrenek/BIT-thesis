from dataclasses import dataclass, field
import os


@dataclass
class DatasetDownloadConfig:
    repo_url: str = (
        f"https://user:{os.getenv("HUGGING_FACE_TOKEN", "")}@huggingface.co/datasets/ShapeNet/ShapeNetCore"
    )
    local_dir: str = "data/raw/ShapeNetCore"


@dataclass
class DatasetConfig:
    name: str = "ShapeNetSynthetic"
    extracted_dir: str = "data/extracted/ShapeNetCore"
    download: DatasetDownloadConfig = field(default_factory=DatasetDownloadConfig)
