from abc import ABC, abstractmethod
from typing import Any

from .dimension import Dimension


class Metric(ABC):
    """Abstract base class for all metrics."""

    name: str
    abbreviation: str
    higher_is_better: bool

    expected_dimensions: tuple[Dimension, ...]

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> float | list[float]:
        """Compute the metric score."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def __str__(self) -> str:
        """Full text representation of the metric including its name and abbreviation
        and parameters for reproducible reporting."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def fingerprint(self) -> dict[str, Any]:
        """Return a dictionary that uniquely identifies the metric and its parameters
        for reproducibility."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _arrow_indicating_optimum(self) -> str:
        """Show an arrow indicating whether higher metric values are better.

        :return: "↑" if higher is better, "↓" otherwise.
        :rtype: str

        """
        return "↑" if self.higher_is_better else "↓"


class MetricList:
    """A list of metrics."""

    def __init__(self, list_name: str = "", metrics: list[Metric] | None = None) -> None:
        self.list_name = list_name
        self.metrics = metrics if metrics is not None else []

    def __call__(self, *args: Any, **kwargs: Any) -> list[float | list[float]]:
        return [m(*args, **kwargs) for m in self.metrics]

    def __str__(self) -> str:
        return f"MetricList(name={self.list_name}, metrics={[str(m) for m in self.metrics]})"

    def fingerprint(self) -> dict[str, Any]:
        return {
            "list_name": self.list_name,
            "metrics": [m.fingerprint() for m in self.metrics],
        }
