from enum import Enum, auto


class Dimension(Enum):
    """Enum for image dimensions."""

    BATCH = auto()
    CHANNEL = auto()
    DEPTH = auto()
    HEIGHT = auto()
    WIDTH = auto()


_DIMENSION_LOOKUP = {
    "B": Dimension.BATCH,
    "b": Dimension.BATCH,
    "C": Dimension.CHANNEL,
    "c": Dimension.CHANNEL,
    "D": Dimension.DEPTH,
    "d": Dimension.DEPTH,
    "H": Dimension.HEIGHT,
    "h": Dimension.HEIGHT,
    "W": Dimension.WIDTH,
    "w": Dimension.WIDTH,
}

__all__ = ["Dimension", "_DIMENSION_LOOKUP"]
