# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import _DIMENSION_LOOKUP, Dimension

logger = get_logger()


def check_image_type(image: np.ndarray | torch.Tensor) -> bool:
    """Check if the image is of a supported type (numpy array or torch tensor).

    :param image: Image to be checked
    :type image: np.ndarray | torch.Tensor
    :return: True if the image type is supported
    :rtype: bool
    :raises TypeError: If the image type is not supported

    """
    if not isinstance(image, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Unsupported image type: {type(image)}. Expected np.ndarray or torch.Tensor.")
    return True


def check_same_type(image: np.ndarray | torch.Tensor, reference: np.ndarray | torch.Tensor) -> bool:
    """Check if the image and reference are of the same type.

    :param image: Image to be checked
    :type image: np.ndarray | torch.Tensor
    :param reference: Reference image to compare type with
    :type reference: np.ndarray | torch.Tensor
    :return: True if the image and reference are of the same type
    :rtype: bool
    :raises TypeError: If the image and reference types do not match

    """
    if type(image) is not type(reference):
        raise TypeError(f"Image type {type(image)} and reference type {type(reference)} do not match.")
    return True


def check_same_shape(image: np.ndarray | torch.Tensor, reference: np.ndarray | torch.Tensor) -> bool:
    """Check if the image and reference have the same shape.

    :param image: Image to be checked
    :type image: np.ndarray | torch.Tensor
    :param reference: Reference image to compare shape with
    :type reference: np.ndarray | torch.Tensor
    :return: True if the image and reference have the same shape
    :rtype: bool
    :raises ValueError: If the image and reference shapes do not match

    """
    if image.shape != reference.shape:
        raise ValueError(f"Image shape {image.shape} and reference shape {reference.shape} do not match.")
    return True


def check_nan_values(image: np.ndarray | torch.Tensor, reference: bool = False) -> bool:
    """Check if the image contains NaN or infinite values. Set reference=True if
    checking a reference image.

    :param image: Image to be checked
    :type image: np.ndarray | torch.Tensor
    :param reference: Whether the image is a reference image
    :type reference: bool
    :return: False if no NaN values are found
    :rtype: bool
    :raises ValueError: If NaN values are found in the image
    :raises TypeError: If the input type is not supported

    """

    label = "Reference image" if reference else "Image"

    if isinstance(image, np.ndarray):
        if np.isnan(image).any():
            raise ValueError(f"{label} contains NaN values.")
        if not np.isfinite(image).all():
            raise ValueError(f"{label} contains non-finite values.")
    elif isinstance(image, torch.Tensor):
        if torch.isnan(image).any():
            raise ValueError(f"{label} contains NaN values.")
        if not torch.isfinite(image).all():
            raise ValueError(f"{label} contains non-finite values.")
    else:
        raise TypeError(f"Input must be a numpy array or a torch tensor, but got {type(image)}.")
    return False


def check_dimensions(dims: Sequence[str], image: np.ndarray | torch.Tensor) -> bool:
    # test that number of specified dimensions matches image dimensions
    if len(dims) != image.ndim:
        raise ValueError(f"Number of specified dimensions {len(dims)} does not match image dimensions {image.ndim}.")

    # test that specified dimensions are valid
    for dim in dims:
        if dim not in _DIMENSION_LOOKUP:
            raise ValueError(f"Invalid dimension '{dim}' specified. Valid dimensions are {_DIMENSION_LOOKUP.keys()}.")

    # test that specified dimensions are unique
    if len(set(dims)) != len(dims):
        raise ValueError(f"Specified dimensions {dims} must be unique, i.e., no duplicates are allowed.")

    return True


def check_value_range(image: torch.Tensor, min_value: float, max_value: float, reference: bool = False) -> bool:
    """Check if the pixel values in the image are within the specified range
    [min_value, max_value].

    :param image: Image to be checked
    :type image: torch.Tensor
    :param min_value: Minimum allowed pixel value
    :type min_value: float
    :param max_value: Maximum allowed pixel value
    :type max_value: float
    :param reference: Whether the image is a reference image (used for error message)
    :type reference: bool
    :return: True if all pixel values are within the specified range
    :rtype: bool
    :raises ValueError: If any pixel value is outside the specified range

    """

    if torch.any(image < min_value) or torch.any(image > max_value):
        raise ValueError(
            f"{'Reference image' if reference else 'Image'} contains pixel values outside "
            f"the range [{min_value}, {max_value}]. Got min {image.min()} and max {image.max()}."
        )
    return True


def warn_if_all_pixels_in_0_to_1_range(image: torch.Tensor, reference: torch.Tensor) -> bool:
    """Warn if all pixel values in both image and reference are in the range [0, 1],
    which may indicate that the images are not correctly scaled for metrics that expect
    pixel values in a different range (e.g., [0, 255]).

    :param image: The input image for which the metric is being computed.
    :type image: torch.Tensor
    :param reference: The reference image to compare against.
    :type reference: torch.Tensor
    :return: True if the warning was issued, False otherwise.
    :rtype: bool

    """
    if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
        logger.warning(
            "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
            "This may indicate that the image is not correctly scaled for this metric, "
            "which expects pixel values in a different range."
        )

    return True


def check_rgb(
    expected_dimensions: tuple[Dimension, ...],
    image: torch.Tensor,
    metric_abbreviation: str,
    additional_error_message: str = "",
) -> None:
    """Check if the image has 3 channels, which is required for RGB images and metrics
    defined on them.

    :param expected_dimensions: The expected dimensions of the metric, used to identify the channel dimension.
    :type expected_dimensions: tuple[Dimension, ...]
    :param image: The input image to be checked.
    :type image: torch.Tensor
    :param metric_abbreviation: The abbreviation of the metric, used for error messages.
    :type metric_abbreviation: str
    :param additional_error_message: Additional error message to be appended to the main error message,
      default is an empty string.
    :type additional_error_message: str
    :raises ValueError: If the image does not have 3 channels,
    which is required for RGB images and metrics defined on them.
    :return: None

    """

    channel_dim = expected_dimensions.index(Dimension.CHANNEL)
    if image.shape[channel_dim] != 3:
        raise ValueError(
            f"Images have {image.shape[channel_dim]} channels, but {metric_abbreviation} expects 3 (RGB) channels."
            f"{additional_error_message}"
        )
