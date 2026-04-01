from typing import Callable

import torch

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.no_reference.base import NoReferenceMetric


class Test(NoReferenceMetric):
    """Test Metric which always returns 0."""

    @property
    def name(self) -> str:
        return "Test Metric"

    @property
    def abbreviation(self) -> str:
        return "TEST"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self) -> None:
        """Initialize Test metric."""
        super().__init__()

    def _compute(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the Test Metric of image."""
        return torch.tensor(0.0)

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the MAE metric."""
        return {}

    def __str__(self) -> str:
        """Full text representation of the Test Metric."""
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"
