"""
Logging utility for Fizix Mech.

Provides a centralized logging configuration for the application.
"""

import logging
import sys

# Configure logger
logger = logging.getLogger("fizixmech")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler if not already present
if not logger.handlers:
    logger.addHandler(console_handler)


def get_logger(name: str = "fizixmech") -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def log_error(message: str, exc: Exception | None = None) -> None:
    """Log an error message with optional exception details."""
    if exc:
        logger.error(f"{message}: {exc}", exc_info=True)
    else:
        logger.error(message)


def log_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(message)


def log_info(message: str) -> None:
    """Log an info message."""
    logger.info(message)


def log_debug(message: str) -> None:
    """Log a debug message."""
    logger.debug(message)
