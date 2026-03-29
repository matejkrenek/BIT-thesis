import sys

from loguru import logger


def _bootstrap_logger() -> None:
    # Ensure consistent single sink even if module is imported multiple times.
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time}</green> <blue>{level}</blue> <level>{message}</level>",
    )


_bootstrap_logger()

__all__ = ["logger"]
