import numpy as np
import pytest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from mondAI.metrics.full_reference import FULL_REFERENCE_METRICS
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.no_reference import NO_REFERENCE_METRICS
from mondAI.metrics.no_reference.base import NoReferenceMetric


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS)
def test_regression_phantom_flip(
    data_regression: DataRegressionFixture, phantom: np.ndarray, metric_class: type[FullReferenceMetric]
) -> None:
    metric = metric_class()
    flipped = np.flip(phantom).copy()
    result = metric(phantom, flipped, dims=("H", "W"))
    data_regression.check({metric.abbreviation: result})


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_regression_phantom(
    data_regression: DataRegressionFixture, phantom: np.ndarray, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()
    result = metric(phantom, dims=("H", "W"))
    data_regression.check({metric.abbreviation: result})


@pytest.mark.parametrize("metric_class", FULL_REFERENCE_METRICS)
def test_regression_brain_flipped(
    ndarrays_regression: NDArraysRegressionFixture, brain: np.ndarray, metric_class: type[FullReferenceMetric]
) -> None:
    metric = metric_class()
    flipped = np.flip(brain, axis=2).copy()
    result = metric(brain, flipped, dims=("D", "H", "W"))
    ndarrays_regression.check({metric.abbreviation: result})


@pytest.mark.parametrize("metric_class", NO_REFERENCE_METRICS)
def test_regression_brain(
    ndarrays_regression: NDArraysRegressionFixture, brain: np.ndarray, metric_class: type[NoReferenceMetric]
) -> None:
    metric = metric_class()
    result = metric(brain, dims=("D", "H", "W"))
    ndarrays_regression.check({metric.abbreviation: result})
