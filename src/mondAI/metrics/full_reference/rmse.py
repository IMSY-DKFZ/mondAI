# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.monai import get_monai_rmse
from mondAI.metrics.third_party.sewar import get_sewar_rmse
from mondAI.metrics.third_party.sklearn import get_sklearn_rmse
from mondAI.metrics.third_party.tensorflow import get_tensorflow_rmse
from mondAI.metrics.third_party.torchmetrics import get_torchmetrics_rmse

logger = get_logger()


class RMSE(FullReferenceMetric):
    r"""Root Mean Squared Error (RMSE).

    The Root Mean Squared Error (RMSE) is a measure of errors between paired images
    expressed as the square root of the average squared difference between the given and reference pixel
    values. It is defined mathematically as:

    .. math::
        \operatorname {RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 }

    where:

    * \\(y_i\\) is the reference pixel value,
    * \\(\\hat{y}_i \\) is the given pixel value,
    * \\(N\\) is the number of pixels per image.

    """

    @property
    def name(self) -> str:
        return "Root Mean Squared Error"

    @property
    def abbreviation(self) -> str:
        return "RMSE"

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
        """Initialize RMSE metric."""
        super().__init__()

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the Root Mean Squared Error between image and reference."""
        return torch.sqrt(torch.mean((image - reference) ** 2))

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(implementations, "scikit-learn", get_sklearn_rmse())
        self._register_implementation(implementations, "tensorflow", get_tensorflow_rmse())
        self._register_implementation(implementations, "torchmetrics", get_torchmetrics_rmse())
        self._register_implementation(implementations, "monai", get_monai_rmse())
        self._register_implementation(implementations, "sewar", get_sewar_rmse())

    def __str__(self) -> str:
        """Full text representation of the RMSE metric."""
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"
