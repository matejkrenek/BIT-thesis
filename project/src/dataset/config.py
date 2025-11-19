from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DatasetDownloadConfig:
    local_dir: Path
    repo_url: Optional[str] = None
    repo_id: str = ""
    token: str = ""


@dataclass
class DatasetConfig:
    local_dir: Path
    download: DatasetDownloadConfig
    categories: list[str]
    name: str = ""
