from collections.abc import Sequence

import numpy as np
import torch

from mondAI.metrics.dimension import _DIMENSION_LOOKUP


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
    """Check if the image contains NaN values. Set reference=True if checking a
    reference image.

    :param image: Image to be checked
    :type image: np.ndarray | torch.Tensor
    :param reference: Whether the image is a reference image
    :type reference: bool
    :return: False if no NaN values are found
    :rtype: bool
    :raises ValueError: If NaN values are found in the image
    :raises TypeError: If the input type is not supported

    """
    if isinstance(image, np.ndarray):
        if np.isnan(image).any():
            raise ValueError(f"{'Reference image' if reference else 'Image'} contains NaN values.")
    elif isinstance(image, torch.Tensor):
        if torch.isnan(image).any():
            raise ValueError(f"{'Reference image' if reference else 'Image'} contains NaN values.")
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
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
