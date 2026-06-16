from importlib.metadata import version

from .logging import get_logger

__version__ = version("mondAI")


logger = get_logger()
logger.info(
    "\n"
    "**********************************************************\n"
    "                   CITATION REMINDER                      \n"
    "**********************************************************\n"
    "If you use any of the metrics implemented in the mondAI   \n"
    "library for your research, please consider citing this    \n"
    "repository and leaving a 🌟 in addition to citing the     \n"
    "original papers that introduced those metrics.            \n"
    "Proper citation helps acknowledge the work of the         \n"
    "researchers who developed these metrics and supports      \n"
    "the continued advancement of the field.\n"
    "**********************************************************\n"
)
