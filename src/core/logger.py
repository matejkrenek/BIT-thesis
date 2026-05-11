"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: logger.py
Responsibility: Configures and exposes the shared Loguru logger instance for project scripts.
"""

import sys

from loguru import logger


def _bootstrap_logger() -> None:
    """Configure a single stdout sink for consistent project logging."""

    # Ensure consistent single sink even if module is imported multiple times.
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time}</green> <blue>{level}</blue> <level>{message}</level>",
    )


_bootstrap_logger()

__all__ = ["logger"]
