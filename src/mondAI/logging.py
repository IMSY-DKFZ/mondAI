import logging
import sys

from mondAI.settings import settings

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

        logger.setLevel(getattr(logging, settings.logging_level))

        _logger = logger

    return _logger
