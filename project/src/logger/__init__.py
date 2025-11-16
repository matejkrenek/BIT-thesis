from loguru import logger
import sys

logger.remove()

logger.add(
    "logs/log-{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="10 days",
    format="{time} {level} {message}",
)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> <blue>{level}</blue> <level>{message}</level>",
)
