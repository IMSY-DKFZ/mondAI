from typing import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric

logger = get_logger()


class MAE(FullReferenceMetric):
    r"""Mean Absolute Error (MAE) metric implementation.

    The Mean Absolute Error (MAE) is a measure of errors between paired observations
    expressed as the average absolute difference between the predicted values and the
    actual values. It is defined mathematically as:

    .. math::
        \operatorname {MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|

    where:

    * \\(y_i\\) is the actual value,
    * \\(\\hat{y}_i \\) is the predicted value,
    * \\(n\\) is the number of observations.

    """

    name = "Mean Absolute Error"
    abbreviation = "MAE"
    higher_is_better = False

    expected_dimensions = (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self) -> None:
        """Initialize MAE metric."""
        super().__init__()

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the Mean Absolute Error between image and reference."""
        return torch.mean(torch.abs(image - reference))

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the MAE metric."""

        implementations = {}

        ### sklearn ###
        try:
            from sklearn.metrics import mean_absolute_error

            def sklearn_mae(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # Flatten the tensors and convert to numpy arrays for sklearn
                image_np = image.cpu().numpy().flatten()
                reference_np = reference.cpu().numpy().flatten()
                mae_value = mean_absolute_error(reference_np, image_np)
                return torch.tensor(mae_value, device=image.device)

            implementations["sklearn"] = sklearn_mae

        except Exception:
            logger.warning("sklearn is not available, skipping sklearn implementation of MAE.")

        ### tensorflow ###
        try:
            import tensorflow as tf

            def tensorflow_mae(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # Flatten the tensors and convert to numpy arrays for TensorFlow

                image_np = image.cpu().numpy()
                reference_np = reference.cpu().numpy()
                mae_metric = tf.keras.metrics.MeanAbsoluteError()
                mae_metric.update_state(reference_np, image_np)
                mae_value = mae_metric.result().numpy()
                return torch.as_tensor(mae_value, device=image.device)

            implementations["tensorflow"] = tensorflow_mae

        except Exception:
            logger.warning("tensorflow is not available, skipping tensorflow implementation of MAE.")

        ### monai ###
        try:
            from monai.metrics import MAEMetric

            def monai_mae(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                mae_metric = MAEMetric()
                mae_value = mae_metric(image.unsqueeze(0), reference.unsqueeze(0))
                return torch.as_tensor(mae_value, device=image.device)

            implementations["monai"] = monai_mae

        except Exception:
            logger.warning("monai is not available, skipping monai implementation of MAE.")

        return implementations

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
