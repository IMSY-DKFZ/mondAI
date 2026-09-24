# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import partial

import torch

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.medimetrics import get_medimetrics_msssim
from mondAI.metrics.third_party.monai import get_monai_msssim
from mondAI.metrics.third_party.piq import get_piq_msssim
from mondAI.metrics.third_party.piqa import get_piqa_msssim
from mondAI.metrics.third_party.sewar import get_sewar_msssim
from mondAI.metrics.third_party.tensorflow import get_tensorflow_msssim
from mondAI.metrics.third_party.torchmetrics import get_torchmetrics_msssim
from mondAI.utils.checks import check_value_range, warn_if_all_pixels_in_0_to_1_range
from mondAI.utils.signal_processing import subsample
from mondAI.utils.similarity_map import ssim_and_cs_maps


class MSSSIM(FullReferenceMetric):
    r"""Multi-Scale Structural SIMilarity (MS-SSIM) index.

    The Multi-Scale Structural Similarity (MS-SSIM) index extends SSIM by
    evaluating image similarity across multiple scales. At each scale, the images are
    compared using local contrast and structure statistics, and then downsampled
    for the next coarser scale. The final score combines the contrast-structure terms
    from the coarser scales with the full SSIM score (including luminance) from the final
    scale.

    This implementation follows the original MATLAB function ``msssim.m`` by Zhou
    Wang as closely as possible in both style and functionality: it uses the same
    Gaussian analysis window, symmetric-padding downsampling with a 2x2 averaging
    filter, and the original weighted product aggregation by default.

    For the default ``product`` method, the score is defined as

    .. math::
        \operatorname{MS\text{-}SSIM}(x, y) =
        \prod_{j=1}^{M-1} \operatorname{MCS}_j(x, y)^{\alpha_j}
        \cdot \operatorname{SSIM}_M(x, y)^{\alpha_M},

    where :math:`M` is the number of scales, :math:`\alpha_j` are the scale
    weights, :math:`\operatorname{MCS}_j` is the mean contrast-structure term at
    scale :math:`j`, and :math:`\operatorname{SSIM}_M` is the mean SSIM value at the
    final scale.

    The per-scale SSIM map is based on

    .. math::
        \operatorname{SSIM}(x, y) =
        \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
        {(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},

    and the contrast-structure component is

    .. math::
        \operatorname{CS}(x, y) = \frac{2\sigma_{xy} + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}.

    Implementation adapted from the original MATLAB implementation by Zhou Wang
    (https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip) and structured
    analogously to the local SSIM implementation in this package.

    Original publication:
    Z. Wang, E. P. Simoncelli, and A. C. Bovik,
    "Multi-scale structural similarity for image quality assessment",
    Proceedings of the 37th Asilomar Conference on Signals, Systems and Computers,
    Nov. 2003, pp. 1398-1402, doi: 10.1109/ACSSC.2003.1292216.

    Differences to some other implementations:

    - this implementation is grayscale-only,
    - it uses convolution with valid padding for the per-scale SSIM computation, matching the
      original MATLAB implementation,
    - it uses symmetric-padding downsampling with a 2x2 averaging filter between
      scales,
    - it supports both the original weighted ``product`` aggregation and the
      alternative ``weighted sum`` mode from the MATLAB reference code,
    - it expects images in ``[0, dynamic_range]`` and defaults to ``dynamic_range=255``.

    """

    DEFAULT_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    @property
    def name(self) -> str:
        return "Multi-Scale Structural Similarity Index"

    @property
    def abbreviation(self) -> str:
        return "MS-SSIM"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def scaling_factor(self) -> float:
        return 255.0

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self,
        k1: float = 0.01,
        k2: float = 0.03,
        kernel_size: int = 11,
        kernel_sigma: float = 1.5,
        dynamic_range: float = 255.0,
        scales: int = 5,
        weights: tuple[float, ...] = DEFAULT_WEIGHTS,
        method: str = "product",
    ) -> None:
        """Initialize MS-SSIM with defaults matching the original implementation.

        :param k1: First stability constant coefficient, default is 0.01, must be non-negative.
        :type k1: float
        :param k2: Second stability constant coefficient, default is 0.03, must be non-negative.
        :type k2: float
        :param kernel_size: Size of the Gaussian window, default is 11, must be odd and at least 3.
        :type kernel_size: int
        :param kernel_sigma: Standard deviation of the Gaussian window, default is 1.5, must be positive.
        :type kernel_sigma: float
        :param dynamic_range: Dynamic range ``L`` of the images, default is 255.0, must be positive.
        :type dynamic_range: float
        :param scales: Number of scales, needs to be a positive integer,
         setting this to 1 results in scores equal to SSIM, default is 5.
        :type scales: int
        :param weights: Weights for each scale, default to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
          which were derived emperically by the original authors.
          The length of weights must match the number of scales, and they will be normalized.
        :type weights: tuple[float, ...]
        :param method: Aggregation method, either 'product' or 'weighted sum', default is 'product'.
        :type method: str
        :raises ValueError: If any of the parameters are out of their valid ranges such as negative values for k1 or k2,
          even kernel size, non-positive kernel sigma or dynamic range, scales less than 1, mismatched weights length,
            zero-sum weights, or invalid method choice.

        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.dynamic_range = dynamic_range
        self.scales = scales
        self.weights = weights
        self.method = method

        if self.k1 < 0 or self.k2 < 0:
            raise ValueError(f"k1 and k2 must be non-negative, but got {self.k1} and {self.k2}.")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {self.kernel_size}.")
        if self.kernel_size * self.kernel_size < 4:
            raise ValueError(
                f"kernel_size must define a window with at least 4 elements, but got {self.kernel_size**2}."
            )
        if self.kernel_sigma <= 0:
            raise ValueError(f"kernel_sigma must be positive, but got {self.kernel_sigma}.")
        if self.dynamic_range <= 0:
            raise ValueError(f"dynamic_range must be positive, but got {self.dynamic_range}.")
        if self.scales < 1:
            raise ValueError(f"scales must be at least 1, but got {self.scales}.")
        if len(self.weights) != self.scales:
            raise ValueError(
                f"weights must have the same length as scales, but got {len(self.weights)} not equal to {self.scales}."
            )
        if sum(self.weights) == 0:
            raise ValueError(f"weights must not sum to zero, but got {sum(self.weights)}.")
        if self.method not in ("product", "weighted sum"):
            raise ValueError(f"method must be either 'product' or 'weighted sum', but got {self.method}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute MS-SSIM between image and reference."""
        self._input_checks(image, reference)

        mean_ssims = image.new_zeros((self.scales,))
        mean_contrasts_and_structures = image.new_zeros((self.scales,))

        for scale in range(self.scales):
            ssim_map, cs_map = ssim_and_cs_maps(
                image,
                reference,
                kernel_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
                dynamic_range=self.dynamic_range,
                k1=self.k1,
                k2=self.k2,
            )
            mean_ssims[scale], mean_contrasts_and_structures[scale] = ssim_map.mean(), cs_map.mean()

            image = subsample(image, kernel_size=2)
            reference = subsample(reference, kernel_size=2)

        weights = torch.tensor(self.weights, device=image.device, dtype=image.dtype)
        terms = torch.cat((mean_contrasts_and_structures[:-1], mean_ssims[-1:]))

        if self.method == "product":
            return torch.prod(terms**weights)

        elif self.method == "weighted sum":
            normalized_weights = weights / weights.sum()
            return torch.sum(terms * normalized_weights)

        else:
            return None  # This should never happen due to the check in __init__

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(
            implementations,
            "torchmetrics",
            partial(
                get_torchmetrics_msssim,
                self.kernel_sigma,
                self.kernel_size,
                self.dynamic_range,
                self.k1,
                self.k2,
                self.weights,
            ),
        )
        self._register_implementation(
            implementations,
            "tensorflow",
            partial(
                get_tensorflow_msssim,
                self.dynamic_range,
                self.weights,
                self.kernel_size,
                self.kernel_sigma,
                self.k1,
                self.k2,
            ),
        )
        self._register_implementation(
            implementations,
            "piq",
            partial(
                get_piq_msssim, self.kernel_size, self.kernel_sigma, self.dynamic_range, self.weights, self.k1, self.k2
            ),
        )
        self._register_implementation(
            implementations,
            "piqa",
            partial(
                get_piqa_msssim, self.kernel_size, self.kernel_sigma, self.dynamic_range, self.weights, self.k1, self.k2
            ),
        )
        self._register_implementation(
            implementations,
            "sewar",
            partial(get_sewar_msssim, self.weights, self.kernel_size, self.k1, self.k2, self.dynamic_range),
        )
        self._register_implementation(
            implementations,
            "monai",
            partial(
                get_monai_msssim,
                self.dynamic_range,
                self.kernel_size,
                self.kernel_sigma,
                self.weights,
                self.k1,
                self.k2,
            ),
        )
        self._register_implementation(
            implementations,
            "medimetrics",
            partial(
                get_medimetrics_msssim,
                self.dynamic_range,
                self.kernel_size,
                self.k1,
                self.k2,
                self.kernel_sigma,
                self.weights,
            ),
        )

    def __str__(self) -> str:
        """Full text representation of the metric."""
        arrow = self._arrow_indicating_optimum()
        return (
            f"{self.name} ({self.abbreviation}) {arrow} with parameters: "
            f"{self.k1=}, {self.k2=}, {self.kernel_size=}, {self.kernel_sigma=}, "
            f"{self.dynamic_range=}, {self.scales=}, {self.weights=}, {self.method=}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to MS-SSIM."""
        check_value_range(image, 0, self.dynamic_range)
        check_value_range(reference, 0, self.dynamic_range, reference=True)

        if self.dynamic_range > 1.0:
            warn_if_all_pixels_in_0_to_1_range(image, reference)

        if image.shape[-2] < self.kernel_size or image.shape[-1] < self.kernel_size:
            raise ValueError(
                f"Images have spatial dimensions {image.shape[-2:]} which are smaller than the required window size "
                f"{self.kernel_size}x{self.kernel_size} for MS-SSIM."
            )

        minimum_image_width = min(image.shape[-2], image.shape[-1]) / (2 ** (self.scales - 1))
        if minimum_image_width < self.kernel_size:
            raise ValueError(
                "Images become too small for the requested number of MS-SSIM scales and window size after repeated "
                "downsampling."
            )
