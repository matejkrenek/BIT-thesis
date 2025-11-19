import dotenv

dotenv.load_dotenv()
from datasets.builder import DatasetBuilder
from datasets.config import DatasetConfig, DatasetDownloadConfig
import os

if __name__ == "__main__":
    config = DatasetConfig()
    builder = DatasetBuilder(config)
    builder.download(force=True).extract(force=True).prepare().syntehsize()
