from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.deepinv import get_deepinv_psnr
from mondAI.metrics.third_party.medimetrics import get_medimetrics_psnr
from mondAI.metrics.third_party.monai import get_monai_psnr
from mondAI.metrics.third_party.piq import get_piq_psnr
from mondAI.metrics.third_party.piqa import get_piqa_psnr
from mondAI.metrics.third_party.sewar import get_sewar_psnr
from mondAI.metrics.third_party.skimage import get_skimage_psnr
from mondAI.metrics.third_party.tensorflow import get_tensorflow_psnr
from mondAI.metrics.third_party.torchmetrics import get_torchmetrics_psnr

logger = get_logger()


class PSNR(FullReferenceMetric):
    r"""Peak Signal-to-Noise Ratio (PSNR).

    Peak Signal-to-Noise Ratio (PSNR) is a full-reference fidelity measure
    derived from the mean squared error (MSE). It expresses the ratio between the
    maximum possible signal value and the reconstruction error on a logarithmic decibel
    scale, ranging from 0 to infinity. Higher PSNR values indicate smaller reconstruction errors,
    and identical images (MSE is zero) yield an infinite PSNR value.

    The metric is defined as

    .. math::
        \operatorname{PSNR}(x, y) = 10 \log_{10}\left(\frac{L^2}{\operatorname{MSE}(x, y)}\right),

    where

    .. math::
        \operatorname{MSE}(x, y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2,

    :math:`L` is the dynamic range of the image intensities, and :math:`N` is the
    number of pixels. Note that :math:`L` is not necessarily the maximum pixel intensity value in the
    given images, but the maximum possible intensity value in the given dynamic range.
    This implementation expects images in ``[0, dynamic_range]`` and defaults to ``dynamic_range=255``.

    Reference implementation used for comparison and API alignment:
    (https://github.com/scikit-image/scikit-image/blob/main/src/skimage/metrics/simple_metrics.py
    commit: d0b36ad1651ee2d7f2a46a84b06ba593a0885c6e, License: BSD-3-Clause).

    """

    @property
    def name(self) -> str:
        return "Peak Signal-to-Noise Ratio"

    @property
    def abbreviation(self) -> str:
        return "PSNR"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, dynamic_range: float = 255.0) -> None:
        """Initialize PSNR.

        :param dynamic_range: Dynamic range ``L`` of the images, which is given by the difference between the maximum
        and minimum possible values (255.0 for unit8, 1.0 for normalized images), default is 255.0, must be positive.
        :type dynamic_range: float
        :raises ValueError: If ``dynamic_range`` is not positive.

        """
        super().__init__()
        self.dynamic_range = dynamic_range

        if self.dynamic_range <= 0:
            raise ValueError(f"dynamic_range must be positive, but got {self.dynamic_range}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute PSNR between image and reference."""
        self._input_checks(image, reference)

        mse = torch.mean((image - reference) ** 2)
        if torch.isclose(mse, torch.tensor(0.0, device=mse.device, dtype=mse.dtype)):
            return torch.tensor(torch.inf, device=image.device, dtype=image.dtype)

        return 10.0 * torch.log10((self.dynamic_range**2) / mse)

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(implementations, "scikit-image", get_skimage_psnr(self.dynamic_range))
        self._register_implementation(implementations, "torchmetrics", get_torchmetrics_psnr(self.dynamic_range))
        self._register_implementation(implementations, "tensorflow", get_tensorflow_psnr(self.dynamic_range))
        self._register_implementation(implementations, "piq", get_piq_psnr(self.dynamic_range))
        self._register_implementation(implementations, "piqa", get_piqa_psnr(self.dynamic_range))
        self._register_implementation(implementations, "sewar", get_sewar_psnr(self.dynamic_range))
        self._register_implementation(implementations, "monai", get_monai_psnr(self.dynamic_range))
        self._register_implementation(implementations, "deepinv", get_deepinv_psnr(self.dynamic_range))
        self._register_implementation(implementations, "medimetrics", get_medimetrics_psnr(self.dynamic_range))

    def __str__(self) -> str:
        """Full text representation of the PSNR metric."""
        arrow = self._arrow_indicating_optimum()
        return f"{self.name} ({self.abbreviation}) {arrow} with {self.dynamic_range=}"

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to PSNR."""
        if torch.any(image < 0) or torch.any(image > self.dynamic_range):
            raise ValueError(f"Input image contains pixel values outside the range [0, {self.dynamic_range}].")

        if torch.any(reference < 0) or torch.any(reference > self.dynamic_range):
            raise ValueError(f"Reference image contains pixel values outside the range [0, {self.dynamic_range}].")

        if (
            torch.all(image >= 0)
            and torch.all(image <= 1)
            and torch.all(reference >= 0)
            and torch.all(reference <= 1)
            and self.dynamic_range > 1.0
        ):
            logger.warning(
                "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "PSNR defaults to dynamic_range=255. Please ensure that your input images are correctly scaled or "
                "set dynamic_range=1.0 for normalized inputs."
            )
