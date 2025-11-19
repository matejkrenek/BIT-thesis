import zipfile
from pathlib import Path
from logger import logger
from .base import BaseExtractor


class ShapeNetExtractor(BaseExtractor):
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def _already_extracted(self) -> bool:
        return self.output_dir.exists() and any(self.output_dir.iterdir())

    def extract(self, force: bool = False):
        if self._already_extracted() and not force:
            logger.info(f"Extraction skipped: {self.output_dir} already contains data.")
            return True

        self.output_dir.mkdir(parents=True, exist_ok=True)

        zip_files = [p for p in self.input_dir.rglob("*.zip") if p.is_file()]
        if not zip_files:
            logger.warning(f"No ZIP archives found in {self.input_dir}.")
            return False

        logger.info(f"Found {len(zip_files)} ZIP archives to extract.")

        for zip_path in zip_files:
            zip_output_dir = self.output_dir / zip_path.stem
            zip_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting {zip_path.name} → {zip_output_dir}")

            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(zip_output_dir)

            logger.success(f"Extracted {zip_path.name}")

        return True
