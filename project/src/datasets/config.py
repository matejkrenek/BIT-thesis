from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and names."""

    name: str = "ShapeNetSyntetic"
    input_path: str = "data/raw"
    output_path: str = "data/syntetic"
