import torch

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.no_reference.base import NoReferenceMetric


class Test(NoReferenceMetric):
    """Test Metric which always returns 0."""

    name = "Test Metric"
    abbreviation = "TEST"
    higher_is_better = False

    expected_dimensions = (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self) -> None:
        """Initialize Test metric."""
        super().__init__()

    def _compute(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the Test Metric of image."""
        return torch.tensor(0.0)

    def __str__(self) -> str:
        """Full text representation of the Test Metric."""
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

    def fingerprint(self) -> dict[str, str | bool]:
        """Return a dictionary that uniquely identifies the Test Metric."""
        return {
            "name": self.name,
            "abbreviation": self.abbreviation,
            "higher_is_better": self.higher_is_better,
        }
