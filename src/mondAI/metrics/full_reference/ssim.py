from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.deepinv import get_deepinv_ssim
from mondAI.metrics.third_party.medimetrics import get_medimetrics_ssim
from mondAI.metrics.third_party.monai import get_monai_ssim
from mondAI.metrics.third_party.piq import get_piq_ssim
from mondAI.metrics.third_party.piqa import get_piqa_ssim
from mondAI.metrics.third_party.sewar import get_sewar_ssim
from mondAI.metrics.third_party.skimage import get_skimage_ssim
from mondAI.metrics.third_party.tensorflow import get_tensorflow_ssim
from mondAI.metrics.third_party.torchmetrics import get_torchmetrics_ssim
from mondAI.utils.signal_processing import subsample
from mondAI.utils.similarity_map import ssim_and_cs_maps

logger = get_logger()


class SSIM(FullReferenceMetric):
    r"""Structural SIMilarity (SSIM) Index.

    The Structural SIMilarity (SSIM) index compares two images by combining
    local luminance, contrast, and structure comparisons. It computes local means,
    variances, and covariance using a sliding window and then averages the resulting
    local SSIM map to obtain the final score. The metric is defined for grayscale
    images and expects pixel values in the range ``[0, L]``, where ``L`` is the
    dynamic range.

    This implementation follows the original MATLAB function ``ssim_index.m`` by
    Zhou Wang as closely as possible in both style and functionality: it uses valid
    convolution, normalizes the analysis window to unit sum, and reproduces the
    branch handling for the special case where one or both stability constants are
    zero.

    The local SSIM map in its specific form (alpha = beta = gamma = 1) is defined as

    .. math::
        \operatorname{SSIM}(x, y) =
        \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
        {(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},

    where :math:`\mu_x` and :math:`\mu_y` are local means, :math:`\sigma_x^2` and
    :math:`\sigma_y^2` are local variances, and :math:`\sigma_{xy}` is the local
    covariance. The constants are

    .. math::
        C_1 = (K_1 L)^2, \qquad C_2 = (K_2 L)^2.

    The final score is the mean over the SSIM map, "MSSIM" in the original paper.

    Implementation adapted from the original MATLAB implementation by Zhou Wang
    (https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m).

    Original publication:
    Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
    "Image quality assessment: From error visibility to structural similarity,"
    IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004,
    doi: 10.1109/TIP.2003.819861.

    Differences to some other implementations:

    - this implementation is grayscale-only,
    - it uses convolution with valid padding exactly like the original MATLAB code, so the local
      SSIM map is smaller than the input image,
    - it expects images in ``[0, dynamic_range]`` and defaults to ``dynamic_range=255``.

    """

    @property
    def name(self) -> str:
        return "Structural Similarity Index Measure"

    @property
    def abbreviation(self) -> str:
        return "SSIM"

    @property
    def higher_is_better(self) -> bool:
        return True

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
        downsample: bool = False,
    ) -> None:
        """Initialize SSIM with defaults matching the original implementation.

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
        :param downsample: Whether to downsample the input images as suggested by the authors for large images.
        The original implementation did not use downsampling but was also only meant for signle scale images.
        If True, the images will be adaptively downsampled to a scale where the smaller dimension is
        approximately 256 pixels, as recommended by the authors for large images. Default is False.
        :type downsample: bool
        :raises ValueError: If any of the parameters are out of their valid ranges such as negative values for k1 or k2,
          even kernel size, kernel size too small, non-positive kernel_sigma or dynamic_range.

        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.dynamic_range = dynamic_range
        self.downsample = downsample
        if self.k1 < 0 or self.k2 < 0:
            raise ValueError(f"k1 and k2 must be non-negative, but got {self.k1} and {self.k2}.")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}.")
        if self.kernel_size * self.kernel_size < 4:
            raise ValueError(
                f"kernel_size must define a window with at least 4 elements, but got {self.kernel_size**2}."
            )
        if self.kernel_sigma <= 0:
            raise ValueError(f"kernel_sigma must be positive, but got {self.kernel_sigma}.")
        if self.dynamic_range <= 0:
            raise ValueError(f"dynamic_range must be positive, but got {self.dynamic_range}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between image and reference."""
        self._input_checks(image, reference)

        # Downsample the images
        if self.downsample:
            min_dimension = min(image.shape)
            scaling_factor = max(1, round(min_dimension / 256))
            image = subsample(image, kernel_size=scaling_factor)
            reference = subsample(reference, kernel_size=scaling_factor)

        ssim_map, _ = ssim_and_cs_maps(
            image,
            reference,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            dynamic_range=self.dynamic_range,
            k1=self.k1,
            k2=self.k2,
        )

        return ssim_map.mean()  # TODO: Also expose the raw SSIM map (and possibly gradients; cf. scikit-image).

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(
            implementations,
            "scikit-image",
            get_skimage_ssim(self.kernel_size, self.dynamic_range, self.k1, self.k2, self.kernel_sigma),
        )
        self._register_implementation(
            implementations,
            "torchmetrics",
            get_torchmetrics_ssim(self.kernel_sigma, self.kernel_size, self.dynamic_range, self.k1, self.k2),
        )
        self._register_implementation(
            implementations,
            "tensorflow",
            get_tensorflow_ssim(self.dynamic_range, self.kernel_size, self.kernel_sigma, self.k1, self.k2),
        )
        self._register_implementation(
            implementations,
            "piq",
            get_piq_ssim(self.kernel_size, self.kernel_sigma, self.dynamic_range, self.k1, self.k2),
        )
        self._register_implementation(
            implementations,
            "piqa",
            get_piqa_ssim(self.kernel_size, self.kernel_sigma, self.dynamic_range, self.k1, self.k2),
        )
        self._register_implementation(
            implementations,
            "sewar",
            get_sewar_ssim(self.kernel_size, self.k1, self.k2, self.dynamic_range, self.kernel_sigma),
        )
        self._register_implementation(
            implementations,
            "monai",
            get_monai_ssim(self.dynamic_range, self.kernel_size, self.kernel_sigma, self.k1, self.k2),
        )
        self._register_implementation(
            implementations,
            "deepinv",
            get_deepinv_ssim(self.dynamic_range, self.kernel_sigma, self.kernel_size, self.k1, self.k2),
        )
        self._register_implementation(
            implementations,
            "medimetrics",
            get_medimetrics_ssim(self.dynamic_range, self.kernel_size, self.kernel_sigma, self.k1, self.k2),
        )

    def __str__(self) -> str:
        """Full text representation of the metric."""
        arrow = self._arrow_indicating_optimum()
        return (
            f"{self.name} ({self.abbreviation}) {arrow} with parameters: "
            f"{self.k1=}, {self.k2=}, {self.kernel_size=}, {self.kernel_sigma=}, "
            f"{self.dynamic_range=}, {self.downsample=}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to SSIM."""
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
                "SSIM defaults to dynamic_range=255. Please ensure that your input images are correctly scaled or "
                "set dynamic_range=1.0 for normalized inputs."
            )

        if image.shape[-2] < self.kernel_size or image.shape[-1] < self.kernel_size:
            raise ValueError(
                f"Images have spatial dimensions {image.shape[-2:]} which are smaller than the required window size "
                f"{self.kernel_size}x{self.kernel_size} for SSIM with valid convolution."
            )
