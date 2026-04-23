"""Tests the `check_*` functions in `utils/checks.py`."""

import re
from collections.abc import Callable, Sequence

import numpy as np
import pytest
import torch

from mondAI.utils.checks import (
    check_dimensions,
    check_image_type,
    check_nan_values,
    check_same_shape,
    check_same_type,
)

### check_nan_values tests ###


@pytest.mark.parametrize(
    "image",
    [
        np.array([[1, 2], [3, 4]]),
        torch.tensor([[1, 2], [3, 4]]),
    ],
)
@pytest.mark.parametrize("reference", [False, True])
def test_check_nan_values_valid(image: np.ndarray | torch.Tensor, reference: bool) -> None:
    assert check_nan_values(image, reference=reference) is False


@pytest.mark.parametrize(
    "image",
    [np.array([[1, 2], [np.nan, 4]]), torch.tensor([[1, 2], [float("nan"), 4]])],
)
@pytest.mark.parametrize("reference", [False, True])
def test_check_nan_values_with_nan(image: np.ndarray | torch.Tensor, reference: bool) -> None:
    with pytest.raises(ValueError, match=f"{'Reference image' if reference else 'Image'} contains NaN values."):
        check_nan_values(image, reference=reference)


@pytest.mark.parametrize(
    "image",
    [None, "invalid_type"],
)
@pytest.mark.parametrize("reference", [False, True])
def test_check_nan_values_with_invalid_input(image: None | str, reference: bool) -> None:
    image = None
    with pytest.raises(
        TypeError,
        match=re.escape(f"Input must be a numpy array or a torch tensor, but got <class '{type(image).__name__}'>."),
    ):
        check_nan_values(image, reference=reference)


### check_image_type tests ###


@pytest.mark.parametrize("image", [np.array([[1, 2], [3, 4]]), torch.tensor([[1, 2], [3, 4]])])
def test_check_image_type_valid(image: np.ndarray | torch.Tensor) -> None:
    assert check_image_type(image) is True


@pytest.mark.parametrize("image", [None, 123, [1, 2, 3], "invalid_type"])
def test_check_image_type_invalid(image: np.ndarray | torch.Tensor) -> None:
    with pytest.raises(TypeError, match=f"Unsupported image type: {type(image)}. Expected np.ndarray or torch.Tensor."):
        check_image_type(image)


### check_dimensions tests ###


@pytest.mark.parametrize(
    "shape, dimensions",
    [
        ((64, 64), ("H", "W")),
        ((64, 64, 3), ("H", "W", "C")),
        ((10, 64, 64, 3), ("B", "H", "W", "C")),
        ((5, 10, 64, 64, 3), ("B", "D", "H", "W", "C")),
    ],
)
@pytest.mark.parametrize("image_generator", [np.zeros, torch.zeros])
def test_check_dimensions_valid(
    shape: tuple[int, ...],
    dimensions: Sequence[str],
    image_generator: Callable[[tuple[int, ...]], np.ndarray | torch.Tensor],
) -> None:
    image = image_generator(shape)
    assert check_dimensions(dimensions, image) is True


def test_check_invalid_dimension_string() -> None:
    image = np.zeros((64, 64))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid dimension 'X' specified. Valid dimensions are dict_keys(['B', 'b', 'C', 'c', 'D', 'd', 'H', 'h',"
            " 'W', 'w'])."
        ),
    ):
        check_dimensions(("H", "X"), image)


def test_check_dimensions_invalid_num_dims() -> None:
    image = np.zeros((64, 64))
    with pytest.raises(ValueError, match=r"Number of specified dimensions 3 does not match image dimensions 2\."):
        check_dimensions(("H", "W", "C"), image)


def test_check_dimensions_invalid_num_dims_singular_dim() -> None:
    image = np.zeros((1, 64, 64))
    with pytest.raises(ValueError, match=r"Number of specified dimensions 2 does not match image dimensions 3\."):
        check_dimensions(("H", "W"), image)


@pytest.mark.parametrize("image", [None, 123, [1, 2, 3], "invalid_type"])
def test_check_dimensions_invalid_type(image: None | int | list[int] | str) -> None:
    with pytest.raises(AttributeError):
        check_dimensions(("H", "W"), image)


def test_check_non_unique_dimensions() -> None:
    image = np.zeros((64, 64, 64))
    with pytest.raises(
        ValueError,
        match=re.escape("Specified dimensions ('W', 'H', 'W') must be unique, i.e., no duplicates are allowed."),
    ):
        check_dimensions(("W", "H", "W"), image)


### check_same_shape tests ###


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 3), (10, 64, 64, 3)])
@pytest.mark.parametrize("image_generator", [np.zeros, torch.zeros])
def test_check_same_shape_numpy_valid(
    shape: tuple[int, ...], image_generator: Callable[[tuple[int, ...]], np.ndarray | torch.Tensor]
) -> None:
    image1 = image_generator(shape)
    image2 = image_generator(shape)
    assert check_same_shape(image1, image2) is True


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 3), (10, 64, 64, 3)])
@pytest.mark.parametrize("image_generator", [np.zeros, torch.zeros])
def test_check_same_shape_numpy_invalid(
    shape: tuple[int, ...], image_generator: Callable[[tuple[int, ...]], np.ndarray | torch.Tensor]
) -> None:
    image1 = image_generator(shape)
    # Modify the last dimension to create a shape mismatch
    shape_mismatch = list(shape)
    shape_mismatch[-1] -= 1
    shape = tuple(shape_mismatch)
    image2 = image_generator(shape)
    with pytest.raises(ValueError, match=r"Image shape .* and reference shape .* do not match\."):
        check_same_shape(image1, image2)


### check_same_type tests ###


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 3), (10, 64, 64, 3)])
@pytest.mark.parametrize("image_generator", [np.zeros, torch.zeros])
def test_check_same_type_valid(
    shape: tuple[int, ...], image_generator: Callable[[tuple[int, ...]], np.ndarray | torch.Tensor]
) -> None:
    image1 = image_generator(shape)
    image2 = image_generator(shape)
    assert check_same_type(image1, image2) is True


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 3), (10, 64, 64, 3)])
def test_check_same_type_invalid(shape: tuple[int, ...]) -> None:
    image1 = np.zeros(shape)
    image2 = torch.zeros(shape)
    with pytest.raises(TypeError, match=r"Image type .* and reference type .* do not match\."):
        check_same_type(image1, image2)
