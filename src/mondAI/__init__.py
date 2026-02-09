from importlib.metadata import version
from typing import Any

from .settings import settings as _settings

__version__ = version("mondAI")


# Forward attribute access to the settings singleton
def __getattr__(name: str) -> Any:
    if hasattr(_settings, name):
        return getattr(_settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __setattr__(name: str, value: object) -> Any:
    if hasattr(_settings, name):
        setattr(_settings, name, value)


# Expose reset_defaults directly
reset_defaults = _settings.reset_to_defaults
