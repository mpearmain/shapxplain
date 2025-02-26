"""
Utility functions for data processing, result formatting, and logging.
"""

import logging
from typing import Optional, Any

# Configure logger
logger = logging.getLogger("shapxplain")


def setup_logger(
    level: int = logging.INFO, log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure the shapxplain logger.

    Args:
        level: Logging level (default: logging.INFO)
        log_format: Custom log format string (optional)

    Returns:
        logging.Logger: Configured logger instance
    """
    if not log_format:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def preprocess_data(data: Any) -> Any:
    """
    Preprocess data for SHAP explanation.

    Args:
        data: Input data to preprocess

    Returns:
        Preprocessed data
    """
    # Placeholder function for preprocessing
    logger.debug("Preprocessing data")
    return data
