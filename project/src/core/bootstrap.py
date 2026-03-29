from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
import open3d as o3d


@dataclass(frozen=True)
class BootstrapConfig:
    device: torch.device
    seed: int
    output_dir: Path
    data_dir: Path
    checkpoint_dir: Path


def bootstrap(
    seed: int = 42,
    *,
    env_file: str | Path | None = None,
    data_subdir: str = "ShapeNetV2",
    checkpoint_subdir: str | None = None,
) -> BootstrapConfig:
    """Load environment defaults and return common script runtime configuration."""
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    load_dotenv(dotenv_path=env_file)

    project_root = Path(os.getenv("ROOT_DIR", ".")).expanduser().resolve()
    output_dir = (
        Path(os.getenv("OUTPUT_DIR", str(project_root / "outputs")))
        .expanduser()
        .resolve()
    )
    data_root = (
        Path(os.getenv("DATA_DIR", str(project_root / "data"))).expanduser().resolve()
    )
    checkpoint_root = (
        Path(os.getenv("CHECKPOINT_DIR", str(project_root / "checkpoints")))
        .expanduser()
        .resolve()
    )

    data_dir = data_root / data_subdir if data_subdir else data_root
    checkpoint_dir = (
        checkpoint_root / checkpoint_subdir if checkpoint_subdir else checkpoint_root
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return BootstrapConfig(
        device=device,
        seed=seed,
        output_dir=output_dir,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
    )
