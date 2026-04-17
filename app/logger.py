import os
import sys

from loguru import logger

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()

# Console logging
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time}</green> | <level>{level}</level> | {message}",
)

# File logging
logger.add(
    f"{LOG_DIR}/app.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time} | {level} | {name}:{function}:{line} | {message}",
)


def get_logger():
    return logger
