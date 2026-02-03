from .base import BaseDownloader
from logger import logger
import zipfile
import requests
import os
from typing import Optional, Callable
from pathlib import Path
import shutil
from urllib.parse import urlparse
from tqdm import tqdm


class ZipUrlDownloader(BaseDownloader):
    """
    Downloader for zip files from URLs with optional postprocessing callback.

    Args:
        local_dir (str): Local directory to save and extract the dataset.
        remote_dir (str): URL of the zip file to download.
        remote_file (Optional[str]): Not used in this implementation (kept for interface compatibility).
        token (Optional[str]): Optional authorization token for HTTP requests.
        force (bool): Whether to force re-download if the dataset already exists locally.
        postprocess_callback (Optional[Callable[[Path], None]]): Optional callback function to reorganize
            the extracted files. The callback receives the extraction directory as parameter.
        chunk_size (int): Size of chunks for downloading large files (default: 8192 bytes).
    """

    def __init__(
        self,
        local_dir: str,
        remote_dir: str,
        remote_file: Optional[str] = None,
        token: Optional[str] = None,
        force: bool = False,
        postprocess_callback: Optional[Callable[[Path], None]] = None,
        chunk_size: int = 8192,
    ):
        super().__init__(local_dir, remote_dir, remote_file, token, force)
        self.postprocess_callback = postprocess_callback
        self.chunk_size = chunk_size
        self.url = remote_dir

    def _get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL or generate one."""
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # If no filename in URL or doesn't end with .zip, generate one
        if not filename or not filename.lower().endswith(".zip"):
            filename = "download.zip"

        return filename

    def _download_file(self, url: str, local_path: Path) -> bool:
        """Download file from URL with progress tracking."""
        try:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            logger.info(f"[ZipUrlDownloader] Starting download from: {url}")

            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Set up tqdm progress bar
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {local_path.name}",
                disable=total_size == 0,  # Disable if size unknown
            )

            with open(local_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            progress_bar.close()

            logger.info(f"[ZipUrlDownloader] Download completed: {local_path}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"[ZipUrlDownloader] Download failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[ZipUrlDownloader] Unexpected error during download: {e}")
            return False

    def _extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract zip file to specified directory."""
        try:
            logger.info(f"[ZipUrlDownloader] Extracting {zip_path} to {extract_to}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Get list of files to be extracted
                file_list = zip_ref.namelist()
                logger.info(f"[ZipUrlDownloader] Extracting {len(file_list)} files...")

                # Extract all files
                zip_ref.extractall(extract_to)

            logger.info(f"[ZipUrlDownloader] Extraction completed to: {extract_to}")
            return True

        except zipfile.BadZipFile:
            logger.error(f"[ZipUrlDownloader] Invalid zip file: {zip_path}")
            return False
        except Exception as e:
            logger.error(f"[ZipUrlDownloader] Error extracting zip file: {e}")
            return False

    def _apply_postprocessing(self, extract_dir: Path) -> bool:
        """Apply postprocessing callback if provided."""
        if self.postprocess_callback:
            try:
                logger.info("[ZipUrlDownloader] Applying postprocessing callback...")
                self.postprocess_callback(extract_dir)
                logger.info("[ZipUrlDownloader] Postprocessing completed successfully")
                return True
            except Exception as e:
                logger.error(f"[ZipUrlDownloader] Postprocessing failed: {e}")
                return False
        return True

    def download(self) -> bool:
        """Download and extract zip file from URL with optional postprocessing."""

        # Check if dataset already exists locally
        if (
            os.path.exists(self.local_dir)
            and os.listdir(self.local_dir)
            and not self.force
        ):
            logger.info("[ZipUrlDownloader] Dataset already exists, skipping download.")
            return True

        try:
            # Create temporary filename for download
            filename = self._get_filename_from_url(self.url)
            temp_zip_path = self.local_dir / filename

            # Ensure local directory exists
            self.local_dir.mkdir(parents=True, exist_ok=True)

            # Download the zip file
            if not self._download_file(self.url, temp_zip_path):
                return False

            # Extract the zip file
            if not self._extract_zip(temp_zip_path, self.local_dir):
                # Clean up downloaded file on extraction failure
                temp_zip_path.unlink(missing_ok=True)
                return False

            # Remove the zip file after successful extraction
            temp_zip_path.unlink(missing_ok=True)
            logger.info(
                f"[ZipUrlDownloader] Removed temporary zip file: {temp_zip_path}"
            )

            # Apply postprocessing if callback is provided
            if not self._apply_postprocessing(self.local_dir):
                logger.warning(
                    "[ZipUrlDownloader] Postprocessing failed, but extraction was successful"
                )
                # Don't return False here as the main download/extraction succeeded

            logger.info(
                f"[ZipUrlDownloader] Dataset successfully downloaded to: {self.local_dir}"
            )
            return True

        except Exception as e:
            logger.error(
                f"[ZipUrlDownloader] Unexpected error during download process: {e}"
            )
            return False


# Example postprocessing callbacks


def flatten_single_directory(extract_dir: Path) -> None:
    """
    Postprocessing callback to flatten directory structure if there's only one subdirectory.
    Useful when zip contains a single root folder with all content inside.
    """
    subdirs = [item for item in extract_dir.iterdir() if item.is_dir()]

    if len(subdirs) == 1:
        single_subdir = subdirs[0]
        temp_dir = extract_dir.parent / f"{extract_dir.name}_temp"

        # Move the single subdirectory to temp location
        shutil.move(str(single_subdir), str(temp_dir))

        # Move all contents from temp to extract_dir
        for item in temp_dir.iterdir():
            shutil.move(str(item), str(extract_dir / item.name))

        # Remove the empty temp directory
        temp_dir.rmdir()


def reorganize_by_file_type(extract_dir: Path) -> None:
    """
    Postprocessing callback to reorganize files by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    files = [item for item in extract_dir.rglob("*") if item.is_file()]

    for file_path in files:
        if file_path.parent == extract_dir:  # Only process files in root
            extension = file_path.suffix.lower()
            if extension:
                type_dir = extract_dir / extension[1:]  # Remove the dot
                type_dir.mkdir(exist_ok=True)
                shutil.move(str(file_path), str(type_dir / file_path.name))


def create_custom_reorganizer(structure_map: dict) -> Callable[[Path], None]:
    """
    Create a custom postprocessing callback based on a structure mapping.

    Args:
        structure_map: Dictionary mapping source patterns to destination directories
                      e.g., {"*.jpg": "images", "*.txt": "texts", "data/*": "dataset"}

    Returns:
        Postprocessing callback function
    """
    import fnmatch

    def custom_reorganizer(extract_dir: Path) -> None:
        for pattern, dest_dir in structure_map.items():
            dest_path = extract_dir / dest_dir
            dest_path.mkdir(exist_ok=True)

            # Find matching files/directories
            for item in extract_dir.rglob("*"):
                if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                    shutil.move(str(item), str(dest_path / item.name))

    return custom_reorganizer
