# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.medimetrics import get_medimetrics_mae
from mondAI.metrics.third_party.monai import get_monai_mae
from mondAI.metrics.third_party.sklearn import get_sklearn_mae
from mondAI.metrics.third_party.tensorflow import get_tensorflow_mae

logger = get_logger()


class MAE(FullReferenceMetric):
    r"""Mean Absolute Error (MAE).

    The Mean Absolute Error (MAE) is a measure of errors between paired images
    expressed as the average absolute difference between the given and reference pixel
    values. It is defined mathematically as:

    .. math::
        \operatorname {MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|

    where:

    * \\(y_i\\) is the actual value,
    * \\(\\hat{y}_i \\) is the predicted value,
    * \\(N\\) is the number of observations.

    """

    @property
    def name(self) -> str:
        return "Mean Absolute Error"

    @property
    def abbreviation(self) -> str:
        return "MAE"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def scaling_factor(self) -> float:
        return 1.0

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self) -> None:
        """Initialize MAE metric."""
        super().__init__()

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the Mean Absolute Error between image and reference."""
        return torch.mean(torch.abs(image - reference))

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(implementations, "scikit-learn", get_sklearn_mae())
        self._register_implementation(implementations, "tensorflow", get_tensorflow_mae())
        self._register_implementation(implementations, "monai", get_monai_mae())
        self._register_implementation(implementations, "medimetrics", get_medimetrics_mae())

    def __str__(self) -> str:
        """Full text representation of the MAE metric."""
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"
