"""This file contains tests for the abstract base classes `Metric`,
`FullReferenceMetric`, and `NoReferenceMetric`.

It verifies the following aspects of the metric base classes:
- The abstract base classes `Metric`, `FullReferenceMetric`, and `NoReferenceMetric` cannot be instantiated directly.
- The abstract methods in the base classes raise `NotImplementedError` when called on an instance of a subclass that
 does not implement them.
- The string representation of a metric includes its name, abbreviation, and an arrow indicating whether higher or
 lower values are better.
- The `fingerprint` method returns a dictionary containing the metric's name, abbreviation, and whether higher values
 are better.

"""

from typing import Any, Callable

import numpy as np
import pytest
from pytest import raises

from mondAI.metrics.base import Metric
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.no_reference.base import NoReferenceMetric

METRICS = [*FullReferenceMetric.__subclasses__(), *NoReferenceMetric.__subclasses__()]


@pytest.mark.parametrize("klass", [Metric, FullReferenceMetric, NoReferenceMetric])
def test_abstract_class_not_instanticable(klass: Metric | FullReferenceMetric | NoReferenceMetric) -> None:
    with raises(TypeError):
        klass()  # type: ignore


@pytest.mark.parametrize(
    "method",
    [
        Metric.__call__,
        Metric.__str__,
        Metric.fingerprint,
        Metric._other_implementations,
        FullReferenceMetric.__str__,
        FullReferenceMetric.fingerprint,
        FullReferenceMetric._other_implementations,
        NoReferenceMetric.__str__,
        NoReferenceMetric.fingerprint,
        NoReferenceMetric._other_implementations,
    ],
)
def test_abstract_method_not_callable(
    dummy_full_reference_metric: FullReferenceMetric, method: Callable[..., Any]
) -> None:
    with raises(NotImplementedError):
        method(dummy_full_reference_metric)


def test_metric_abstract_compute_iteratively_method_not_callable(
    dummy_full_reference_metric: FullReferenceMetric, phantom: np.ndarray
) -> None:
    with raises(NotImplementedError):
        Metric._compute_iteratively(dummy_full_reference_metric, phantom, phantom, compute_function=lambda x: x)


def test_metric_abstract_compute_method_not_callable(
    dummy_full_reference_metric: FullReferenceMetric, phantom: np.ndarray
) -> None:
    with raises(NotImplementedError):
        FullReferenceMetric._compute(dummy_full_reference_metric, phantom, phantom)


def test_metric_abstract_no_reference_compute__method_not_callable(
    dummy_no_reference_metric: NoReferenceMetric, phantom: np.ndarray
) -> None:
    with raises(NotImplementedError):
        NoReferenceMetric._compute(dummy_no_reference_metric, phantom)


def test_metric_str_dummy_representation(dummy_full_reference_metric: FullReferenceMetric) -> None:
    assert str(dummy_full_reference_metric) == "Dummy Full Reference Metric (DFRM) ↑"


@pytest.mark.parametrize("metric_class", METRICS)
def test_metric_str_actual_representation(metric_class: type[Metric]) -> None:
    metric = metric_class()
    representation = str(metric)
    assert metric.name in representation
    assert metric.abbreviation in representation
    assert "↑" in representation or "↓" in representation


def test_metric_dummy_fingerprint(dummy_full_reference_metric: FullReferenceMetric) -> None:
    fingerprint = dummy_full_reference_metric.fingerprint()
    assert fingerprint["name"] == "Dummy Full Reference Metric"
    assert fingerprint["abbreviation"] == "DFRM"
    assert fingerprint["higher_is_better"] is True


@pytest.mark.parametrize("metric_class", METRICS)
def test_metric_actual_fingerprint(metric_class: type[Metric]) -> None:
    metric = metric_class()
    fingerprint = metric.fingerprint()
    assert "name" in fingerprint
    assert "abbreviation" in fingerprint
    assert "higher_is_better" in fingerprint
    assert isinstance(fingerprint["name"], str)
    assert isinstance(fingerprint["abbreviation"], str)
    assert isinstance(fingerprint["higher_is_better"], bool)
