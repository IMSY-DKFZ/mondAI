import re

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


def test_check_nan_values_numpy_no_nan() -> None:
    image = np.array([[1, 2], [3, 4]])
    assert check_nan_values(image) is False


def test_check_nan_values_numpy_with_nan() -> None:
    image = np.array([[1, 2], [np.nan, 4]])
    with pytest.raises(ValueError, match="Image contains NaN values."):
        check_nan_values(image)


def test_check_nan_values_with_invalid_none() -> None:
    image = None
    with pytest.raises(TypeError, match="Input must be a numpy array or a torch tensor."):
        check_nan_values(image)


def test_check_nan_values_torch_no_nan() -> None:
    image = torch.tensor([[1, 2], [3, 4]])
    assert check_nan_values(image) is False


def test_check_nan_values_torch_with_nan() -> None:
    image = torch.tensor([[1, 2], [float("nan"), 4]])
    with pytest.raises(ValueError, match="Image contains NaN values."):
        check_nan_values(image)


def test_check_nan_values_invalid_type() -> None:
    image = "invalid_type"
    with pytest.raises(TypeError, match="Input must be a numpy array or a torch tensor."):
        check_nan_values(image)


def test_check_nan_values_reference_numpy_no_nan() -> None:
    image = np.array([[1, 2], [3, 4]])
    assert check_nan_values(image, reference=True) is False


def test_check_nan_values_reference_numpy_with_nan() -> None:
    image = np.array([[1, 2], [np.nan, 4]])
    with pytest.raises(ValueError, match="Reference image contains NaN values."):
        check_nan_values(image, reference=True)


def test_check_nan_values_reference_torch_no_nan() -> None:
    image = torch.tensor([[1, 2], [3, 4]])
    assert check_nan_values(image, reference=True) is False


def test_check_nan_values_reference_torch_with_nan() -> None:
    image = torch.tensor([[1, 2], [float("nan"), 4]])
    with pytest.raises(ValueError, match="Reference image contains NaN values."):
        check_nan_values(image, reference=True)


def test_check_nan_values_reference_invalid_type() -> None:
    image = "invalid_type"
    with pytest.raises(TypeError, match="Input must be a numpy array or a torch tensor."):
        check_nan_values(image, reference=True)


def test_check_image_type_numpy() -> None:
    image = np.array([[1, 2], [3, 4]])
    assert check_image_type(image) is True


def test_check_image_type_torch() -> None:
    image = torch.tensor([[1, 2], [3, 4]])
    assert check_image_type(image) is True


def test_check_image_type_invalid() -> None:
    image = "invalid_type"
    with pytest.raises(TypeError, match="Unsupported image type: <class 'str'>. Expected np.ndarray or torch.Tensor."):
        check_image_type(image)


def test_check_image_type_invalid_int() -> None:
    image = 123
    with pytest.raises(TypeError, match="Unsupported image type: <class 'int'>. Expected np.ndarray or torch.Tensor."):
        check_image_type(image)


def test_check_image_type_invalid_list() -> None:
    image = [1, 2, 3]
    with pytest.raises(TypeError, match="Unsupported image type: <class 'list'>. Expected np.ndarray or torch.Tensor."):
        check_image_type(image)


def test_check_dimensions_valid_2d() -> None:
    image = np.zeros((64, 64))
    assert check_dimensions(("W", "H"), image) is True


def test_check_dimensions_valid_3d() -> None:
    image = np.zeros((64, 64, 3))
    assert check_dimensions(("W", "H", "C"), image) is True


def test_check_dimensions_valid_4d() -> None:
    image = np.zeros((10, 64, 64, 3))
    assert check_dimensions(("B", "W", "H", "C"), image) is True


def test_check_dimensions_valid_5d() -> None:
    image = np.zeros((5, 10, 64, 64, 3))
    assert check_dimensions(("B", "D", "W", "H", "C"), image) is True


def test_check_invalid_dimension_string() -> None:
    image = np.zeros((64, 64))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid dimension 'X' specified. Valid dimensions are dict_keys(['B', 'b', 'C', 'c', 'D', 'd', 'H', 'h',"
            " 'W', 'w'])."
        ),
    ):
        check_dimensions(("W", "X"), image)


def test_check_dimensions_invalid_num_dims() -> None:
    image = np.zeros((64, 64))
    with pytest.raises(ValueError, match="Number of specified dimensions 3 does not match image dimensions 2."):
        check_dimensions(("W", "H", "C"), image)


def test_check_dimensions_invalid_num_dims_single_dim() -> None:
    image = np.zeros((1, 64, 64))
    with pytest.raises(ValueError, match="Number of specified dimensions 2 does not match image dimensions 3."):
        check_dimensions(("W", "H"), image)


def test_check_dimensions_invalid_type() -> None:
    image = "invalid_type"
    with pytest.raises(AttributeError):
        check_dimensions(("W", "H"), image)


def test_check_dimensions_torch_valid_2d() -> None:
    image = torch.zeros((64, 64))
    assert check_dimensions(("W", "H"), image) is True


def test_check_dimensions_torch_invalid_type() -> None:
    image = 123
    with pytest.raises(AttributeError):
        check_dimensions(("W", "H"), image)


def test_check_non_unique_dimensions() -> None:
    image = np.zeros((64, 64, 64))
    with pytest.raises(
        ValueError,
        match=re.escape("Specified dimensions ('W', 'H', 'W') must be unique, i.e., no duplicates are allowed."),
    ):
        check_dimensions(("W", "H", "W"), image)


def test_check_same_shape_numpy_valid() -> None:
    image1 = np.zeros((10, 64, 64, 3))
    image2 = np.zeros((10, 64, 64, 3))
    assert check_same_shape(image1, image2) is True


def test_check_same_shape_numpy_invalid() -> None:
    image1 = np.zeros((10, 64, 64, 3))
    image2 = np.zeros((10, 64, 32, 3))
    with pytest.raises(ValueError, match="Image shape .* and reference shape .* do not match."):
        check_same_shape(image1, image2)


def test_check_same_shape_torch_valid() -> None:
    image1 = torch.zeros((10, 64, 64, 3))
    image2 = torch.zeros((10, 64, 64, 3))
    assert check_same_shape(image1, image2) is True


def test_check_same_shape_torch_invalid() -> None:
    image1 = torch.zeros((10, 64, 64, 3))
    image2 = torch.zeros((10, 64, 32, 3))
    with pytest.raises(ValueError, match="Image shape .* and reference shape .* do not match."):
        check_same_shape(image1, image2)


def test_check_same_type_valid() -> None:
    image1 = np.zeros((10, 64, 64, 3))
    image2 = np.zeros((10, 64, 64, 3))
    assert check_same_type(image1, image2) is True


def test_check_same_type_invalid() -> None:
    image1 = np.zeros((10, 64, 64, 3))
    image2 = torch.zeros((10, 64, 64, 3))
    with pytest.raises(TypeError, match="Image type .* and reference type .* do not match."):
        check_same_type(image1, image2)


def test_check_same_type_valid_torch() -> None:
    image1 = torch.zeros((10, 64, 64, 3))
    image2 = torch.zeros((10, 64, 64, 3))
    assert check_same_type(image1, image2) is True
