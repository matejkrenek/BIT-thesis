from dotenv import load_dotenv
from logger import logger
from pathlib import Path

logger.info("Loading environment variables from .env file...")
load_dotenv(Path(".env"))

from datasets.config import DatasetConfig
from datasets.downloaders.huggingface import HuggingFaceDownloader
import os

logger.info("Environment variables loaded successfully.")

logger.info("Initializing dataset downloader...")
dataset_config = DatasetConfig()

logger.info("Creating Hugging Face downloader...")
downloader = HuggingFaceDownloader(
    repo_id=dataset_config.download.repo_id,
    local_dir=dataset_config.download.local_dir,
    token=dataset_config.download.token,
)

logger.info("Starting dataset download...")
downloader.download()
logger.info("Dataset download completed.")
