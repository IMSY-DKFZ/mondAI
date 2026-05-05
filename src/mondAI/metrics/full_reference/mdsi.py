from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.piq import get_piq_mdsi
from mondAI.metrics.third_party.piqa import get_piqa_mdsi
from mondAI.utils.signal_processing import gradient_map, subsample
from mondAI.utils.similarity_map import similarity_map

logger = get_logger()


class MDSI(FullReferenceMetric):
    r"""Mean Deviation Similarity Index (MDSI)

    This metric computes the mean deviation of a combined gradient-chromaticity similarity map between a distorted image
    and a reference image. It expects RGB images with pixel values in the range [0, 255] and yields scores between 0 and
    infinity, where lower values indicate better perceptual quality. MDSI is designed to capture both gradient and
    color distortions in a way that correlates well with human perception of image quality. It is defined as

    ..math::
        \operatorname{MDSI} = \left( \frac{1}{N} \sum_{i=1}^{N} \left| GCS_i^{\frac{1}{4}} -
        \overline{GCS^{\frac{1}{4}}} \right| \right)^{\frac{1}{4}}

    where :math:`GCS_i` is the combined gradient-chromaticity similarity at location i,
    and :math:`\overline{GCS}` its mean. :math:`GCS` is computed as a weighted combination of the gradient similarity
    and chromaticity similarity components of MDSI, usually as the weighted sum:

    ..math::
    GCS = \alpha \cdot GS + (1 - \alpha) \cdot CS

    where :math:`GS` is the gradient similarity map, :math:`CS` is the chromaticity similarity map, and :math:`\alpha`
    is a weighting factor empirically set to 0.6 in the original paper.
    The gradient similarity component of MDSI is computed based on the gradient magnitudes of the luminance channel of
    the images with a special weighting done by fusing distored and reference images,
    while the chromaticity similarity component is computed based on the opponent color channels. T


    Implementation adapted from original MATLAB implementation by Hossein Ziaei Nafchi, which is available at:
    https://de.mathworks.com/matlabcentral/fileexchange/59809-mdsi-ref-dist-combmethod/files/MDSI.m
    (licence: BSD 2-Clause License)

    Original publication:
    Nafchi, H. Z., Shahkolaei, A., Hedjam, R., & Cheriet, M. (2016).
    Mean deviation similarity index: Efficient and reliable full-reference image quality evaluator.
    IEE Access, 4, 5579-5590.

    """

    @property
    def name(self) -> str:
        return "Mean Deviation Similarity Index"

    @property
    def abbreviation(self) -> str:
        return "MDSI"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, combination_method: str = "sum") -> None:
        """

        :param combination_method: Method to combine the gradient similarity and chromaticity similarity components
        of MDSI. Must be either 'sum' or 'product'. Default is 'sum'.
        :type combination_method: str

        """
        super().__init__()
        self.combination_method = combination_method

        if self.combination_method not in ("sum", "product"):
            raise ValueError(
                f"combination_method must be either 'sum' or 'product', but got {self.combination_method}."
            )

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self._input_checks(image, reference)

        C1 = 140
        C2 = 55
        C3 = 550
        prewitt_kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], device=image.device, dtype=image.dtype) / 3

        # Downsample the images
        min_dimension = min(
            image.shape[self.expected_dimensions.index(dim)] for dim in (Dimension.HEIGHT, Dimension.WIDTH)
        )
        kernel_size = max(1, round(min_dimension / 256))
        image = subsample(image, kernel_size=kernel_size, channels=3)
        reference = subsample(reference, kernel_size=kernel_size, channels=3)

        # Convert to luminance and opponent color channels
        rgb_to_lhm_matrix = torch.tensor(
            [[0.2989, 0.587, 0.114], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]], device=image.device, dtype=image.dtype
        )
        image_lhm = torch.einsum("xc,chw->xhw", rgb_to_lhm_matrix, image)
        reference_lhm = torch.einsum("xc,chw->xhw", rgb_to_lhm_matrix, reference)

        # gradient magnitudes
        gradient_image = gradient_map(image_lhm[0], prewitt_kernel)
        gradient_reference = gradient_map(reference_lhm[0], prewitt_kernel)
        gradient_fusion = gradient_map(0.5 * (image_lhm[0] + reference_lhm[0]), prewitt_kernel)

        # gradient similarity
        gradient_similarity_RD = similarity_map(gradient_image, gradient_reference, C1)
        gradient_similarity_RF = similarity_map(gradient_reference, gradient_fusion, C2)
        gradient_similarity_DF = similarity_map(gradient_image, gradient_fusion, C2)
        gradient_similarity = gradient_similarity_RD + gradient_similarity_DF - gradient_similarity_RF

        # chromaticity similarity
        chromaticity_similarity = (2 * (image_lhm[1] * reference_lhm[1] + image_lhm[2] * reference_lhm[2]) + C3) / (
            image_lhm[1] ** 2 + reference_lhm[1] ** 2 + image_lhm[2] ** 2 + reference_lhm[2] ** 2 + C3
        )

        # convert to complex numbers to handle negative numberes and exponents smaller than 1 without NaNs
        gradient_similarity = gradient_similarity.to(torch.complex128)
        chromaticity_similarity = chromaticity_similarity.to(torch.complex128)

        # gradient-chromaticity similarity
        if self.combination_method == "sum":
            alpha = 0.6
            gradient_chromaticity_similarity = alpha * gradient_similarity + (1 - alpha) * chromaticity_similarity
        elif self.combination_method == "product":
            gamma = 0.2
            beta = 0.1
            gradient_chromaticity_similarity = gradient_similarity**gamma * chromaticity_similarity**beta

        # deviation pooling
        mean_gradient_chromaticity_similarity = torch.mean(gradient_chromaticity_similarity**0.25)
        return (
            torch.mean(torch.abs(gradient_chromaticity_similarity**0.25 - mean_gradient_chromaticity_similarity))
            ** 0.25
        )

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(implementations, "piq", get_piq_mdsi(self.combination_method))
        self._register_implementation(implementations, "piqa", get_piqa_mdsi(self.combination_method))

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) "
            f"{self._arrow_indicating_optimum()} "
            f"with combination_method={self.combination_method}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to MDSI, such as checking for valid pixel
        value ranges and dimensions.

        Warns if all pixel values in both image and reference are in the range [0, 1],
        which may indicate that the images are not correctly scaled for MDSI, which
        expects pixel values in the range [0, 255].

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :raises ValueError: If the input image contains pixel values outside the range
            [0, 255].
        :raises ValueError: If the reference image contains pixel values outside the
            range [0, 255].
        :raises ValueError: If images do not have 3 channels, which is required for
            MDSI.

        """
        # Input checks specific to MDSI
        if torch.any(image < 0) or torch.any(image > 255):
            raise ValueError("Input image contains pixel values outside the range [0, 255].")

        if torch.any(reference < 0) or torch.any(reference > 255):
            raise ValueError("Reference image contains pixel values outside the range [0, 255].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            logger.warning(
                "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "MDSI expects pixel values in the range [0, 255]. Please ensure that your input images are "
                "correctly scaled for accurate computation of MDSI."
            )

        channel_dim = self.expected_dimensions.index(Dimension.CHANNEL)
        if image.shape[channel_dim] != 3:
            raise ValueError(f"Images have {image.shape[channel_dim]} channels. MDSI expects 3 (RGB) channels.")
