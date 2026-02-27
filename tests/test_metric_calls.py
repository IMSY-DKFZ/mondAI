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
from mondAI.metrics.dimension import Dimension
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

            return self._call_pipeline(image, reference, reference, dims=dims, compute_function=self._compute)

        def _compute_iteratively(
            self, *reshaped_images: torch.Tensor, compute_function: Callable[..., torch.Tensor]
        ) -> torch.Tensor:
            images, references = reshaped_images
            return torch.stack(
                [
                    compute_function(image, reference)
                    for image, reference in zip(images.unbind(0), references.unbind(0), strict=True)
                ]
            )

    metric = InvalidMetric()
    with raises(ValueError, match="InvalidMetric supports 1 or 2 input images, got 3."):
        metric(phantom, phantom, dims=("H", "W"))


@pytest.mark.parametrize(
    "image_dimensions, metric_dimensions",
    [
        (("H", "W"), (Dimension.HEIGHT, Dimension.WIDTH)),
        (("D", "H", "W"), (Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH)),
        (("D", "H", "W"), (Dimension.HEIGHT, Dimension.WIDTH)),
        (("H", "W", "D"), (Dimension.HEIGHT, Dimension.WIDTH)),
        (("C", "H", "W"), (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)),
        (("C", "H", "W"), (Dimension.HEIGHT, Dimension.WIDTH)),
        (("C", "H", "W"), (Dimension.CHANNEL,)),
        (("H", "W", "C"), (Dimension.CHANNEL,)),
        (("B", "D", "H", "W"), (Dimension.BATCH, Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "D", "H", "W"), (Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "D", "H", "W"), (Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "C", "H", "W"), (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "C", "H", "W"), (Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "C", "H", "W"), (Dimension.CHANNEL,)),
        (("B", "C", "H", "W"), (Dimension.BATCH,)),
        (
            ("B", "C", "D", "H", "W"),
            (Dimension.BATCH, Dimension.CHANNEL, Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH),
        ),
        (
            ("B", "C", "H", "W", "D"),
            (Dimension.BATCH, Dimension.CHANNEL, Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH),
        ),
        (("B", "C", "D", "H", "W"), (Dimension.CHANNEL, Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "C", "D", "H", "W"), (Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "C", "D", "H", "W"), (Dimension.HEIGHT, Dimension.WIDTH)),
        (("B", "C", "D", "H", "W"), (Dimension.CHANNEL,)),
        (("B", "C", "D", "H", "W"), (Dimension.BATCH,)),
        (("B", "C", "D", "H", "W"), (Dimension.DEPTH,)),
        (("B", "C", "D", "H", "W"), (Dimension.BATCH, Dimension.CHANNEL)),
        # (("H", "W"), ("D", "H", "W")),  # invalid case: image has fewer dimensions than metric expects
        # (("D", "H", "W"), ("H", "W")),  # invalid case: image has more dimensions than metric expects
        # (("C", "H", "W"), ("H", "W")),  # invalid case: image has more dimensions than metric expects
    ],
)
def test_output_shape_depending_on_image_and_metric_dimensions(
    image_dimensions: Sequence[str],
    metric_dimensions: tuple[Dimension, ...],
    output_shape_test_metric_factory: Callable[[tuple[Dimension, ...]], FullReferenceMetric],
) -> None:
    img = torch.rand(tuple(4 for _ in image_dimensions))  # create a random tensor with the specified image dimensions

    metric = output_shape_test_metric_factory(metric_dimensions)
    result = metric(img, img, dims=image_dimensions, compare_implementations=False)

    # check if output shape is correct based on the specified metric dimensions and image dimensions
    expected_output_dims = len(image_dimensions) - len(metric_dimensions)

    if expected_output_dims == 0:
        assert isinstance(result, float), f"Expected result to be a scalar (float), but got {type(result)}"
    elif expected_output_dims > 0:
        assert isinstance(result, torch.Tensor), f"Expected result to be a torch.Tensor, but got {type(result)}"
        assert result.ndim == expected_output_dims, (
            f"Expected result to have {expected_output_dims} dimensions, but got {result.ndim}"
        )
        assert all(dim == 4 for dim in result.shape), f"Expected result shape to contain only 4, but got {result.shape}"
    else:
        raise ValueError("Expected output dimensions cannot be negative.")


@pytest.mark.parametrize(
    "image_dimensions, metric_dimensions",
    [
        (
            ("H", "W"),
            (Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH),
        ),  # invalid case: image has fewer dimensions than metric expects
        (
            ("H", "W"),
            (Dimension.HEIGHT, Dimension.WIDTH, Dimension.CHANNEL),
        ),  # invalid case: image has fewer dimensions than metric expects
        (("H", "W"), (Dimension.CHANNEL,)),  # invalid case: image has other dimension than metric expects
        (
            ("C", "H", "W"),
            (Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH),
        ),  # invalid case: image has other dimension than metric expects
        (
            ("B", "C", "H", "W"),
            (Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH),
        ),  # imvalid case: image has other dimension than metric expects
    ],
)
def test_invalid_image_and_metric_dimensions(
    image_dimensions: Sequence[str],
    metric_dimensions: tuple[Dimension, ...],
    output_shape_test_metric_factory: Callable[[tuple[Dimension, ...]], FullReferenceMetric],
) -> None:
    img = torch.rand(tuple(4 for _ in image_dimensions))  # create a random tensor with the specified image dimensions

    metric = output_shape_test_metric_factory(metric_dimensions)

    with pytest.raises(ValueError):
        metric(img, img, dims=image_dimensions, compare_implementations=False)
