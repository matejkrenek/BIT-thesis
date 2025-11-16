from .config import DatasetConfig
from pathlib import Path
from urllib.request import urlretrieve


class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.raw_path = Path(config.input_path)
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def build(self):
        zip_path = self.download()
        print("✅ Dataset ready at:", self.raw_path.resolve())

    def download(self):
        """Download dataset from the specified URL in the config."""
        zip_path = self.raw_path / f"{self.config.name}.zip"
        if zip_path.exists():
            print("✔ ShapeNet archive already exists, skipping download.")
            return zip_path
        print(f"⬇ Downloading {self.config.name} from {self.config.download_url} ...")
        urlretrieve(self.config.download_url, zip_path)
        print("✅ Download complete:", zip_path)
        return zip_path
