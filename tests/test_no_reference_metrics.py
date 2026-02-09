import re

import numpy as np
import pytest
import torch
from pytest import raises

from mondAI.metrics.no_reference import NO_REFERENCE_METRICS
from mondAI.metrics.no_reference.base import NoReferenceMetric


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_validity(phantom: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    assert metric(phantom) == 0.0


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_without_dimensions_valid(phantom: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    result = metric(phantom)  # assumes dims=("W", "H") by default
    assert isinstance(result, float)


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_dimensions_valid(phantom: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    result = metric(phantom, dims=("W", "H"))
    assert isinstance(result, float)


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_more_than_one_dimension_invalid(
    phantom: np.ndarray, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()
    with raises(ValueError, match="Number of specified dimensions 1 does not match image dimensions 2."):
        metric(phantom, dims=("H",))


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_more_than_two_dimensions_invalid(
    brain: np.ndarray, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()
    with raises(ValueError, match="Number of specified dimensions 2 does not match image dimensions 3."):
        metric(brain, dims=("W", "H"))


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_dimensions_invalid(phantom: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    with raises(ValueError, match="Number of specified dimensions 3 does not match image dimensions 2."):
        metric(phantom, dims=("W,", "H", "D"))


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_non_unique_dimensions(phantom: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    with raises(
        ValueError, match=re.escape("Specified dimensions ('W', 'W') must be unique, i.e., no duplicates are allowed.")
    ):
        metric(phantom, dims=("W", "W"))


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_invalid_dimensions(phantom: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    with raises(
        ValueError,
        match=re.escape(
            "Invalid dimension 'X' specified. Valid dimensions are dict_keys(['B', 'b', 'C', 'c', "
            "'D', 'd', 'H', 'h', 'W', 'w'])."
        ),
    ):
        metric(phantom, dims=("X", "W"))


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_wrong_dimension(phantom: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    with raises(
        ValueError,
        match=re.escape(
            "Expected dimension 'Dimension.HEIGHT' not found in specified dimensions [<Dimension.DEPTH: 3>, "
            "<Dimension.WIDTH: 5>]."
        ),
    ):
        metric(phantom, dims=("D", "W"))


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_without_dimensions_torch_valid(
    grayscale_image_W_H: torch.Tensor, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()
    result = metric(grayscale_image_W_H)  # assumes dims=("W", "H") by default
    assert isinstance(result, float)


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_dimensions_numpy_valid(brain: np.ndarray, metric_class: type[NoReferenceMetric]) -> None:
    metric = metric_class()
    result = metric(brain, dims=("D", "W", "H"))
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_metric_call_with_dimensions_torch_valid(
    rgb_image_C_W_H: torch.Tensor, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()
    result = metric(rgb_image_C_W_H, dims=("C", "W", "H"))
    assert isinstance(result, torch.Tensor)
