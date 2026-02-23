import torch

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric


class MAE(FullReferenceMetric):
    """Mean Absolute Error (MAE) metric implementation.

    The Mean Absolute Error (MAE) is a measure of errors between paired observations
    expressed as the average absolute difference between the predicted values and the
    actual values. It is defined mathematically as:

    \[
    MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
    \]

    where:
    - \(y_i\) is the actual value,
    - \(\hat{y}_i\) is the predicted value,
    - \(n\) is the number of observations.

    """

    name = "Mean Absolute Error"
    abbreviation = "MAE"
    higher_is_better = False

    expected_dimensions = (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self) -> None:
        """Initialize MAE metric."""
        super().__init__()

    def _check_metric_configuration(self) -> bool:
        """Check if the metric configuration is valid."""
        # No specific configuration to check for MAE
        return True

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the Mean Absolute Error between image and reference."""
        return torch.mean(torch.abs(image - reference))

    def __str__(self) -> str:
        """Full text representation of the MAE metric."""
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

    def fingerprint(self) -> dict[str, str | bool]:
        """Return a dictionary that uniquely identifies the MAE metric."""
        return {
            "name": self.name,
            "abbreviation": self.abbreviation,
            "higher_is_better": self.higher_is_better,
        }
