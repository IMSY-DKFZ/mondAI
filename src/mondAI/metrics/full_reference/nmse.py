# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.deepinv import get_deepinv_nmse
from mondAI.metrics.third_party.medimetrics import get_medimetrics_nmse

logger = get_logger()


class NMSE(FullReferenceMetric):
    r"""Normalized Mean Squared Error (NMSE).

    The Normalized Mean Squared Error (NMSE) is a measure of errors between paired images
    expressed as the square root of the average squared difference between the given and reference pixel
    values.

    In most cases, the NMSE is normalized by the squared L2 norm of the reference image.

    It is defined mathematically as:

    .. math::
        \operatorname {NMSE} = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 /  \sum_{i=1}^{N} y_i^2



    where:

    * \\(y_i\\) is the reference pixel value,
    * \\(\\hat{y}_i \\) is the given pixel value,
    * \\(N\\) is the number of pixels per image.

    """

    @property
    def name(self) -> str:
        return "Normalized Mean Squared Error"

    @property
    def abbreviation(self) -> str:
        return "NMSE"

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
        """Initialize NMSE metric."""
        super().__init__()

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the Normalized Mean Squared Error between image and reference."""
        return torch.mean((image - reference) ** 2) / torch.mean(reference**2)

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(implementations, "deepinv", get_deepinv_nmse)
        self._register_implementation(
            implementations, "medimetrics", get_medimetrics_nmse
        )  # normalizes with std of target and not with mean of target, so values are different

    def __str__(self) -> str:
        """Full text representation of the NMSE metric."""
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"
