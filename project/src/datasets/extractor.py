import zipfile
from pathlib import Path
from logger import logger


class Extractor:
    """Extracts ShapeNet synset ZIP packages into dedicated folders."""

    def __init__(self, raw_dir: str = "data/raw", out_dir: str = "data/extracted"):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(self):
        zip_files = list(self.raw_dir.glob("*.zip"))
        if not zip_files:
            logger.warning("No ZIP files found.")
            return
        for zip_path in zip_files:
            self.extract_zip(zip_path)

    def extract_zip(self, zip_path: Path):
        synset_id = zip_path.stem
        target_dir = self.out_dir / synset_id

        if target_dir.exists():
            logger.info(f"Synset {synset_id} already extracted.")
            return

        logger.info(f"Extracting {zip_path.name} -> {target_dir}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.out_dir)
        logger.success(f"Extracted {synset_id}")
