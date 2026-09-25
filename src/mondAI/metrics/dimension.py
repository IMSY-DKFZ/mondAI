# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from enum import Enum, auto


class Dimension(Enum):
    """Enum for image dimensions."""

    BATCH = auto()
    CHANNEL = auto()
    DEPTH = auto()
    HEIGHT = auto()
    WIDTH = auto()


def is_compatible(image_dimensions: Sequence[str], expected_metric_dimensions: Sequence[Dimension]) -> bool:
    """Check if the image dimensions are compatible with the expected metric
    dimensions.

    This method checks if all expected metric dimensions are present in the image
    dimensions. It uses a lookup dictionary to map dimension characters (e.g., 'H',
    'W') to Dimension enums.

    :param image_dimensions: A sequence of dimension characters representing the
        dimensions of the input image (e.g., ('B', 'C', 'H', 'W')).
    :type image_dimensions: Sequence[str]
    :param expected_metric_dimensions: A sequence of Dimension enums representing the
        dimensions expected by the metric.
    :type expected_metric_dimensions: Sequence[Dimension]
    :return: True if all expected metric dimensions are present in the image
        dimensions, False otherwise.
    :rtype: bool

    """
    image_dims_as_enums = {_DIMENSION_LOOKUP.get(dim) for dim in image_dimensions}
    return all(expected_dim in image_dims_as_enums for expected_dim in expected_metric_dimensions)


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


__all__ = ["_DIMENSION_LOOKUP", "Dimension", "is_compatible"]
