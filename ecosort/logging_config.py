"""EcoSort Logging Configuration."""

import logging
from typing import Optional


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure Python logging for EcoSort.

    Args:
        log_level: Logging level (default: logging.INFO).
        log_file: Optional file path to also log to a file.

    Returns:
        The root logger.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt))
        root_logger.addHandler(file_handler)

    return root_logger
