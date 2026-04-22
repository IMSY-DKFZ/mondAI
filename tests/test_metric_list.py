"""This file contains tests for the `MetricList` class defined in `metrics/base.py`.

It verifies the following aspects of the `MetricList` class:
- Initialization of the `MetricList` with default values and with a specified name.
- The string representation of the `MetricList` includes its name and class name.
- The `fingerprint` method returns a dictionary containing the list name and the list of metrics.
- The `__call__` method correctly computes the metrics in the list for both full reference and no reference metrics,
and returns results of the expected length.

"""

import numpy as np
import pytest

from mondAI.metrics.base import Metric, MetricList
from mondAI.metrics.dimension import is_compatible
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.no_reference.base import NoReferenceMetric

METRICS = [*FullReferenceMetric.__subclasses__(), *NoReferenceMetric.__subclasses__()]


def test_metric_list_initialization_default() -> None:
    metric_list = MetricList()
    assert metric_list.list_name == ""
    assert metric_list.metrics == []


def test_metric_list_initialization_with_name() -> None:
    metric_list = MetricList(list_name="test_metrics")
    assert metric_list.list_name == "test_metrics"
    assert metric_list.metrics == []


def test_metric_list_str_representation() -> None:
    metric_list = MetricList(list_name="empty_list")
    assert "empty_list" in str(metric_list)
    assert "MetricList" in str(metric_list)


def test_metric_list_fingerprint() -> None:
    metric_list = MetricList(list_name="test")
    fingerprint = metric_list.fingerprint()
    assert fingerprint["list_name"] == "test"
    assert fingerprint["metrics"] == []


@pytest.mark.parametrize("metric_class", METRICS)
def test_metric_list_call(phantom: np.ndarray, metric_class: type[Metric]) -> None:
    metric = metric_class()
    metric_list = MetricList(metrics=[metric])

    if not is_compatible(("H", "W"), metric.expected_dimensions):
        pytest.skip(
            f"Skipping test for {metric.abbreviation} with expected dimensions "
            f"{metric.expected_dimensions} and phantom image because they are incompatible."
        )

    if issubclass(metric_class, FullReferenceMetric):
        results = metric_list(phantom, phantom)
    elif issubclass(metric_class, NoReferenceMetric):
        results = metric_list(phantom)
    else:
        raise ValueError(f"Unknown metric super class {metric_class}")
    assert len(results) == 1


def test_metric_list_call_empty() -> None:
    metric_list = MetricList()
    results = metric_list()
    assert results == []
