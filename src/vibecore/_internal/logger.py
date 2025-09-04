"""
Logger configuration module for console output.

This module sets up the standard console logger.
"""

import logging
import os

# define defaults/read from environment
LOGGER_PREFIX = "VibeCore"
LOGGER_FORMAT = "[%(name)s] %(levelname)s: %(message)s"
LOGGER_LEVEL = os.getenv("VIBE_LOG_LEVEL", "ERROR").upper()

# ansi colors
LEVEL_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "ERROR": "\033[31m",  # Red
}
YELLOW = "\033[33m"
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """
    Log formatter that adds ANSI colors.

    - Colors the log level name based on LEVEL_COLORS.
    - Colors the logger prefix (name) in yellow/orange.
    """

    def format(self, record: logging.LogRecord):
        """
        Apply color formatting to a log record before output.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message.

        """
        color = LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{RESET}"
        record.name = f"{YELLOW}{record.name}{RESET}"
        return super().format(record)


class ConsoleLogger(logging.Logger):
    """
    Console logger class that configures itself on initialization.
    """

    def __new__(cls, name: str = LOGGER_PREFIX):
        """
        Create a new ConsoleLogger instance.

        Args:
            name (str): The name of the logger.

        Returns:
            ConsoleLogger: The configured logger instance.

        """
        logger = logging.getLogger(name)

        log_level_num = getattr(logging, LOGGER_LEVEL, logging.ERROR)
        logger.setLevel(log_level_num)

        formatter = ColorFormatter(LOGGER_FORMAT)

        if not logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        logger.info("✅ Logger is successfully configured.")
        return logger
