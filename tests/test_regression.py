"""This file contains regression tests.

It validates that computing metrics on the phantom and brain images produces consistent
results. The tests compute each metric on the phantom and brain images (for full
refernce metrics compared to the flipped images), and compare the results to previously
stored values using pytest-regressions. This ensures that any changes to the metric
implementations do not cause unintended changes in the computed values, thus helping to
catch regressions in the codebase.

"""

import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from mondAI.metrics.dimension import is_compatible
from mondAI.metrics.full_reference import FULL_REFERENCE_METRICS
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.no_reference import NO_REFERENCE_METRICS
from mondAI.metrics.no_reference.base import NoReferenceMetric

ROUND_DIGITS = 9


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS)
def test_regression_phantom_flip(
    data_regression: DataRegressionFixture, phantom: np.ndarray, metric_class: type[FullReferenceMetric]
) -> None:
    metric = metric_class()

    if not is_compatible(("H", "W"), metric.expected_dimensions):
        pytest.skip(
            f"Skipping test for {metric.abbreviation} with expected dimensions {metric.expected_dimensions}"
            f"and phantom image because they are incompatible."
        )

    flipped = np.flip(phantom).copy()
    result = metric(phantom, flipped, dims=("H", "W"))
    data_regression.check({metric.abbreviation: result}, round_digits=ROUND_DIGITS)


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_regression_phantom(
    data_regression: DataRegressionFixture, phantom: np.ndarray, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()

    if not is_compatible(("H", "W"), metric.expected_dimensions):
        pytest.skip(
            f"Skipping test for {metric.abbreviation} with expected dimensions {metric.expected_dimensions}"
            f"and phantom image because they are incompatible."
        )

    result = metric(phantom, dims=("H", "W"))
    data_regression.check({metric.abbreviation: result}, round_digits=ROUND_DIGITS)


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS)
def test_regression_brain_flipped(
    ndarrays_regression: NDArraysRegressionFixture, brain: np.ndarray, metric_class: type[FullReferenceMetric]
) -> None:
    metric = metric_class()

    if not is_compatible(("D", "H", "W"), metric.expected_dimensions):
        pytest.skip(
            f"Skipping test for {metric.abbreviation} with expected dimensions {metric.expected_dimensions}"
            f"and brain image because they are incompatible."
        )

    flipped = np.flip(brain, axis=2).copy()
    result = metric(brain, flipped, dims=("D", "H", "W"))
    ndarrays_regression.check({metric.abbreviation: result})


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_regression_brain(
    ndarrays_regression: NDArraysRegressionFixture, brain: np.ndarray, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()

    if not is_compatible(("D", "H", "W"), metric.expected_dimensions):
        pytest.skip(
            f"Skipping test for {metric.abbreviation} with expected dimensions {metric.expected_dimensions}"
            f"and brain image because they are incompatible."
        )

    result = metric(brain, dims=("D", "H", "W"))
    ndarrays_regression.check({metric.abbreviation: result})


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS)
def test_regression_maximum_dimensions_flipped(
    ndarrays_regression: NDArraysRegressionFixture,
    maximum_dimensions_image_B_C_D_H_W: np.ndarray,
    metric_class: type[FullReferenceMetric],
) -> None:
    metric = metric_class()

    if not is_compatible(("B", "C", "D", "H", "W"), metric.expected_dimensions):
        pytest.skip(
            f"Skipping test for {metric.abbreviation} with expected dimensions {metric.expected_dimensions}"
            f"and maximum dimensions image because they are incompatible."
        )

    flipped = np.flip(maximum_dimensions_image_B_C_D_H_W, axis=-2).copy()
    result = metric(maximum_dimensions_image_B_C_D_H_W, flipped, dims=("B", "C", "D", "H", "W"))
    if isinstance(result, float):
        result = np.array(result)  # convert to array for consistent output type for regression testing
    ndarrays_regression.check({metric.abbreviation: result})


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_regression_maximum_dimensions(
    ndarrays_regression: NDArraysRegressionFixture,
    maximum_dimensions_image_B_C_D_H_W: np.ndarray,
    metric_class: type[NoReferenceMetric],
) -> None:
    metric = metric_class()

    if not is_compatible(("B", "C", "D", "H", "W"), metric.expected_dimensions):
        pytest.skip(
            f"Skipping test for {metric.abbreviation} with expected dimensions {metric.expected_dimensions}"
            f"and maximum dimensions image because they are incompatible."
        )

    result = metric(maximum_dimensions_image_B_C_D_H_W, dims=("B", "C", "D", "H", "W"))

    if isinstance(result, float):
        result = np.array(result)  # convert to array for consistent output type for regression testing

    ndarrays_regression.check({metric.abbreviation: result})
