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

    def _check_inputs_for_metric(self, *inputs: torch.Tensor) -> bool:
        """Check if the input images are compatible with the metric configuration.

        :param inputs: A tuple containing the input images and references, typically in
            the form (image, reference).
        :type inputs: tuple[torch.Tensor, torch.Tensor]
        :return: True if the input images and metric are compatible, otherwise raises a
            ValueError.
        :rtype: bool
        :raises ValueError: If the input images are not compatible with the metric
            configuration.

        """
        (image,) = inputs

        # No specific configuration to check for Test Metric

        return True

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
