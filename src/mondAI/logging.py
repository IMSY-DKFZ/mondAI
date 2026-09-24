# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Returns a singleton logger object that wraps around Python's built-in logging
    module. It ensures that all logging in the mondAI package is consistent and can be
    easily configured.

    Usage:

    .. code-block:: python

        from mondAI.logging import get_logger

        logger = get_logger()  # Get the singleton logger instance

        logger.info("This is an info message.")
        logger.debug("This is a debug message.")

        logger.setLevel(logging.DEBUG)  # Change logging level to DEBUG after instantiation
        logger.debug("This debug message will now be shown.")

    Use :code:`settings.logging_level` to set the default logging level for the entire package. This will only affect
    loggers that are instantiated after the change. To change the logging level of an existing logger, use
    :code:`logger.setLevel(logging.DEBUG)` or the appropriate level.

    """
    global _logger

    if _logger is None:
        logger = logging.getLogger("mondAI")

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler("mondAI.log", mode="w")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)

        _logger = logger

    return _logger


def update_log_level(level: str) -> None:
    """Updates the logging level of the singleton logger instance. This function is
    called internally when :code:`settings.logging_level` is updated, but can also be
    used directly to change the logging level of an existing logger.

    Args:
        level (str): The new logging level. Must be one of "DEBUG", "INFO", "WARNING", "ERROR".

    Raises:
        ValueError: If the provided logging level is not valid.

    """
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        raise ValueError(f"logging_level must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', but is {level}")

    logger = get_logger()
    logger.setLevel(getattr(logging, level))
