"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: __init__.py
Responsibility: Exports all dataset downloader classes for various remote sources (HuggingFace, Kaggle, Zip URL).
"""

from .base import BaseDownloader
from .huggingface import HuggingFaceDownloader
from .kaggle import KaggleDownloader
from .zip_url import ZipUrlDownloader

__all__ = [
    "BaseDownloader",
    "HuggingFaceDownloader",
    "KaggleDownloader",
    "ZipUrlDownloader",
]
