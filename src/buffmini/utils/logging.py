"""Logging configuration helpers."""

from __future__ import annotations

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured logger."""

    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    logger.setLevel(level)
    return logger
