from loguru import logger
import sys

# Remove default logger
logger.remove()

# Add stdout logger with custom format
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> <blue>{level}</blue> <level>{message}</level>",
)
