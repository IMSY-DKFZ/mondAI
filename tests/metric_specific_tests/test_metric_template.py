import pytest

from mondAI.metrics.full_reference.metric_template import MetricTemplate


def test_negative_parameter() -> None:
    with pytest.raises(ValueError):
        MetricTemplate(parameter1=-1)
