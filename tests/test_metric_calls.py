"""This file contains tests for the call function of all metrics defined in `metrics`.

It verifies the following aspects of the metrics:
- Valid calls to the metric with and without specifying dimensions.
- Invalid calls to the metric with incorrect dimensions, non-unique dimensions, and invalid dimension names.
- Valid calls to the metric with both numpy arrays and torch tensors, ensuring that the output type matches input type.
- Comparison of the default implementation with other implementations (if available) to ensure consistency in results.
- Checking that the metric raises appropriate exceptions for invalid inputs and dimension specifications.

"""

import re
from typing import Callable, Sequence

import numpy as np
import pytest
import torch
from pytest import raises

from mondAI.metrics.base import Metric
from mondAI.metrics.full_reference import FULL_REFERENCE_METRICS
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.no_reference import NO_REFERENCE_METRICS
from mondAI.metrics.no_reference.base import NoReferenceMetric


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
@pytest.mark.parametrize("image_fixture", ["phantom", "grayscale_image_W_H"])
def test_metric_call_without_dimensions_valid(
    request: pytest.FixtureRequest, metric_class: type[Metric], image_fixture: str
) -> None:
    image = request.getfixturevalue(image_fixture)
    metric = metric_class()
    if isinstance(metric, FullReferenceMetric):
        metric(image, image)  # assumes dims=("W", "H") by default
    elif isinstance(metric, NoReferenceMetric):
        metric(image)  # assumes dims=("W", "H") by default


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
@pytest.mark.parametrize(
    "image_fixture, dimensions",
    [("phantom", ("W", "H")), ("grayscale_image_W_H", ("W", "H")), ("brain", ("D", "H", "W"))],
)
def test_metric_call_with_dimensions_valid(
    request: pytest.FixtureRequest,
    metric_class: type[Metric],
    image_fixture: str,
    dimensions: Sequence[str],
) -> None:
    image = request.getfixturevalue(image_fixture)
    metric = metric_class()
    if isinstance(metric, FullReferenceMetric):
        metric(image, image, dims=dimensions)
    elif isinstance(metric, NoReferenceMetric):
        metric(image, dims=dimensions)


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
@pytest.mark.parametrize("dimensions", [("H",), ("W",), ("D",), ("C",), ("B",), ("b",), ("h",), ("w",)])
def test_metric_call_with_more_than_one_dimension_invalid(
    phantom: np.ndarray, metric_class: type[Metric], dimensions: Sequence[str]
) -> None:
    metric = metric_class()
    with raises(ValueError, match="Number of specified dimensions 1 does not match image dimensions 2."):
        if isinstance(metric, FullReferenceMetric):
            metric(phantom, phantom, dims=dimensions)
        elif isinstance(metric, NoReferenceMetric):
            metric(phantom, dims=dimensions)


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
def test_metric_call_with_more_than_two_dimensions_invalid(brain: np.ndarray, metric_class: type[Metric]) -> None:
    metric = metric_class()
    with raises(ValueError, match="Number of specified dimensions 2 does not match image dimensions 3."):
        if isinstance(metric, FullReferenceMetric):
            metric(brain, brain, dims=("H", "W"))
        elif isinstance(metric, NoReferenceMetric):
            metric(brain, dims=("H", "W"))


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
def test_metric_call_with_dimensions_invalid(phantom: np.ndarray, metric_class: type[Metric]) -> None:
    metric = metric_class()
    with raises(ValueError, match="Number of specified dimensions 3 does not match image dimensions 2."):
        if isinstance(metric, FullReferenceMetric):
            metric(phantom, phantom, dims=("H,", "W", "D"))
        elif isinstance(metric, NoReferenceMetric):
            metric(phantom, dims=("H,", "W", "D"))


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
def test_metric_call_with_non_unique_dimensions(phantom: np.ndarray, metric_class: type[Metric]) -> None:
    metric = metric_class()
    with raises(
        ValueError, match=re.escape("Specified dimensions ('W', 'W') must be unique, i.e., no duplicates are allowed.")
    ):
        if isinstance(metric, FullReferenceMetric):
            metric(phantom, phantom, dims=("W", "W"))
        elif isinstance(metric, NoReferenceMetric):
            metric(phantom, dims=("W", "W"))


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
def test_metric_call_with_invalid_dimensions(phantom: np.ndarray, metric_class: type[Metric]) -> None:
    metric = metric_class()
    with raises(
        ValueError,
        match=re.escape(
            "Invalid dimension 'X' specified. Valid dimensions are dict_keys(['B', 'b', 'C', 'c', "
            "'D', 'd', 'H', 'h', 'W', 'w'])."
        ),
    ):
        if isinstance(metric, FullReferenceMetric):
            metric(phantom, phantom, dims=("X", "W"))
        elif isinstance(metric, NoReferenceMetric):
            metric(phantom, dims=("X", "W"))


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
def test_metric_call_with_wrong_dimension(phantom: np.ndarray, metric_class: type[Metric]) -> None:
    metric = metric_class()
    with raises(
        ValueError,
        match=re.escape(
            "Expected dimension 'Dimension.HEIGHT' not found in specified dimensions [<Dimension.DEPTH: 3>, "
            "<Dimension.WIDTH: 5>]."
        ),
    ):
        if isinstance(metric, FullReferenceMetric):
            metric(phantom, phantom, dims=("D", "W"))
        elif isinstance(metric, NoReferenceMetric):
            metric(phantom, dims=("D", "W"))


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
@pytest.mark.parametrize(
    "image_fixture, dimensions", [("brain", ("D", "H", "W")), ("rgb_image_C_W_H", ("C", "W", "H"))]
)
def test_metric_call_with_dimensions_return_type_valid(
    request: pytest.FixtureRequest,
    metric_class: type[Metric],
    image_fixture: str,
    dimensions: Sequence[str],
) -> None:
    image = request.getfixturevalue(image_fixture)
    metric = metric_class()

    if isinstance(metric, FullReferenceMetric):
        result = metric(image, image, dims=dimensions)
    elif isinstance(metric, NoReferenceMetric):
        result = metric(image, dims=dimensions)

    assert isinstance(result, type(image))  # output type should match input type


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS)
@pytest.mark.parametrize(
    "image_fixture, dimensions",
    [
        ("phantom", ("W", "H")),
        ("grayscale_image_W_H", ("W", "H")),
        ("brain", ("D", "H", "W")),
        ("rgb_image_C_W_H", ("C", "W", "H")),
    ],
)
def test_metric_call_with_compare_implementations(
    request: pytest.FixtureRequest,
    metric_class: type[Metric],
    image_fixture: str,
    dimensions: Sequence[str],
) -> None:
    image = request.getfixturevalue(image_fixture)
    metric = metric_class()

    if isinstance(metric, FullReferenceMetric):
        result = metric(image, image, dims=dimensions, compare_implementations=True)
    elif isinstance(metric, NoReferenceMetric):
        result = metric(image, dims=dimensions, compare_implementations=True)

    assert isinstance(result, dict)
    assert "mondAI" in result
    assert len(result) == len(metric._other_implementations()) + 1  # +1 for the default implementation


def test_metric_call_with_vectorization_invalid_input_number(phantom: np.ndarray) -> None:
    class InvalidMetric(Metric):
        name = "Invalid Metric"
        abbreviation = "IM"
        higher_is_better = True

        def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            return torch.Tensor(0.0)

        def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
            return {}

        def __str__(self) -> str:
            return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

        def fingerprint(self) -> dict[str, str | bool]:
            return {
                "name": self.name,
                "abbreviation": self.abbreviation,
                "higher_is_better": self.higher_is_better,
            }

        def __call__(
            self,
            image: np.ndarray | torch.Tensor,
            reference: np.ndarray | torch.Tensor,
            dims: Sequence[str] = ("H", "W"),
        ) -> float | torch.Tensor | np.ndarray:
            """Compute the metric between image and corresponding two references
            (error)."""

            return self._call_with_vectorization(image, reference, reference, dims=dims, compute_function=self._compute)

        def _compute_vmapped(
            self, *reshaped_images: torch.Tensor, compute_function: Callable[..., torch.Tensor]
        ) -> torch.Tensor:
            images, references = reshaped_images
            return torch.vmap(compute_function)(images, references)

    metric = InvalidMetric()
    with raises(ValueError, match="InvalidMetric supports 1 or 2 input images, got 3."):
        metric(phantom, phantom, dims=("H", "W"))
