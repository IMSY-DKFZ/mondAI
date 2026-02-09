import numpy as np
import pytest
import torch
from pytest import raises

from mondAI.metrics.base import Metric, MetricList
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.no_reference.base import NoReferenceMetric

METRICS = [*FullReferenceMetric.__subclasses__(), *NoReferenceMetric.__subclasses__()]


def test_metric_abstract_class_not_instantiable() -> None:
    with raises(TypeError):
        Metric()  # type: ignore


def test_full_reference_metric_abstract_class_not_instantiable() -> None:
    with raises(TypeError):
        FullReferenceMetric()  # type: ignore


def test_no_reference_metric_abstract_class_not_instantiable() -> None:
    with raises(TypeError):
        NoReferenceMetric()  # type: ignore


def test_metric_str_dummy_representation() -> None:
    class DummyMetric(FullReferenceMetric):
        name = "Dummy Metric"
        abbreviation = "DM"
        higher_is_better = True

        def _check_metric_configuration(self) -> bool:
            return True

        def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> float:
            return 0.0

        def __str__(self) -> str:
            return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

        def fingerprint(self) -> dict[str, str | bool]:
            return {
                "name": self.name,
                "abbreviation": self.abbreviation,
                "higher_is_better": self.higher_is_better,
            }

    metric = DummyMetric()
    assert str(metric) == "Dummy Metric (DM) ↑"


@pytest.mark.parametrize("metric_class", METRICS)
def test_metric_str_representation(metric_class: type[Metric]) -> None:
    metric = metric_class()
    representation = str(metric)
    assert metric.name in representation
    assert metric.abbreviation in representation
    assert "↑" in representation or "↓" in representation


def test_metric_dummy_fingerprint() -> None:
    class DummyMetric(FullReferenceMetric):
        name = "Dummy Metric"
        abbreviation = "DM"
        higher_is_better = True

        def _check_metric_configuration(self) -> bool:
            return True

        def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> float:
            return 0.0

        def __str__(self) -> str:
            return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

        def fingerprint(self) -> dict[str, str | bool]:
            return {
                "name": self.name,
                "abbreviation": self.abbreviation,
                "higher_is_better": self.higher_is_better,
            }

    metric = DummyMetric()
    fingerprint = metric.fingerprint()
    assert fingerprint["name"] == "Dummy Metric"
    assert fingerprint["abbreviation"] == "DM"
    assert fingerprint["higher_is_better"] is True


@pytest.mark.parametrize("metric_class", METRICS)
def test_metric_fingerprint(metric_class: type[Metric]) -> None:
    metric = metric_class()
    fingerprint = metric.fingerprint()
    assert "name" in fingerprint
    assert "abbreviation" in fingerprint
    assert "higher_is_better" in fingerprint
    assert isinstance(fingerprint["name"], str)
    assert isinstance(fingerprint["abbreviation"], str)
    assert isinstance(fingerprint["higher_is_better"], bool)


### MetricList tests ###
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
def test_metric_list_reflexivity(phantom: np.ndarray, metric_class: type[Metric]) -> None:
    metric_list = MetricList(metrics=[metric_class()])
    if issubclass(metric_class, FullReferenceMetric):
        results = metric_list(phantom, phantom)
    elif issubclass(metric_class, NoReferenceMetric):
        results = metric_list(phantom)
    else:
        raise ValueError(f"Unknown metric class {metric_class}")
    assert len(results) == 1
    assert results[0] == 0.0
