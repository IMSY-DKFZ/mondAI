from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.piq import get_piq_mdsi
from mondAI.metrics.third_party.piqa import get_piqa_mdsi
from mondAI.utils.checks import check_rgb, check_value_range, warn_if_all_pixels_in_0_to_1_range
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
    def scaling_factor(self) -> float:
        return 255.0

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self,
        combination_method: str = "sum",
        C1: int = 140,
        C2: int = 55,
        C3: int = 550,
        alpha: float = 0.6,
        beta: float = 0.1,
        gamma: float = 0.2,
        rho: float = 1.0,
        q: float = 0.25,
        o: float = 0.25,
    ) -> None:
        """

        :param combination_method: Method to combine the gradient similarity and chromaticity similarity components
        of MDSI. Must be either 'sum' or 'product'. Default is 'sum'.
        :type combination_method: str
        :param C1: Coefficient to calculate gradient similarity. Default is 140.
        :type C1: int
        :param C2: Coefficient to calculate gradient similarity with fused images. Default is 55.
        :type C2: int
        :param C3: Coefficient to calculate chromaticity similarity. Default is 550.
        :type C3: int
        :param alpha: Coefficient to combine gradient similarity and chromaticity similarity when using summation.
        Should be between 0 and 1. Default is 0.6.
        :type alpha: float
        :param beta: Power to combine gradient similarity with chromaticity similarity when using multiplication.
        Should be a positive value. Default is 0.1.
        :type beta: float
        :param gamma: Power to combine gradient similarity and chromaticity similarity when using multiplication.
         Should be a positive value. Default is 0.2.
        :type gamma: float
        :param rho: Order of the Minkowski distance used in deviation pooling. Should be a positive value.
        Default is 1.0.
        :type rho: float
        :param q: Coefficient to adjust the emphasis of the values in image and mean chromaticity similarity map
        when computing the gradient-chromaticity similarity. Should be a positive value. Default is 0.25.
        :type q: float
        :param o: The power pooling applied on the final value of the deviation. Should be a positive value.
          Default is 0.25.
        :type o: float
        """
        super().__init__()
        self.combination_method = combination_method
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.q = q
        self.o = o

        if self.combination_method not in ("sum", "product"):
            raise ValueError(
                f"combination_method must be either 'sum' or 'product', but got {self.combination_method}."
            )

        if self.C1 != 140 or self.C2 != 55 or self.C3 != 550:
            logger.warning(
                f"Using non-default values of C1={C1}, C2={C2}, or C3={C3} for MDSI may lead to results that are "
                "not directly comparable to the original formulation of MDSI, which uses C1=140, C2=55, and C3=550. "
                "Please ensure that you understand the implications of changing these parameters on the metric's "
                "behavior and interpretability."
            )

        if not (0 <= self.alpha <= 1):
            raise ValueError(f"alpha must be between 0 and 1, but got {self.alpha}.")

        if self.beta <= 0:
            raise ValueError(f"beta must be a positive value, but got {self.beta}.")

        if self.gamma <= 0:
            raise ValueError(f"gamma must be a positive value, but got {self.gamma}.")

        if self.rho <= 0:
            raise ValueError(f"rho must be a positive value, but got {self.rho}.")

        if self.q <= 0:
            raise ValueError(f"q must be a positive value, but got {self.q}.")

        if self.o <= 0:
            raise ValueError(f"o must be a positive value, but got {self.o}.")

        if (
            self.alpha != 0.6
            or self.beta != 0.1
            or self.gamma != 0.2
            or self.rho != 1.0
            or self.q != 0.25
            or self.o != 0.25
        ):
            logger.warning(
                f"Using non-default values of alpha={alpha}, beta={beta}, gamma={gamma}, rho={rho}, q={q}, or o={o} "
                "for MDSI may lead to results that are not directly comparable to the original formulation of MDSI, "
                "which uses alpha=0.6, beta=0.1, gamma=0.2, rho=1.0, q=0.25, and o=0.25. Please ensure that you "
                "understand the implications of changing these parameters on the metric's behavior and "
                "interpretability."
            )

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self._input_checks(image, reference)

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
        gradient_similarity_RD = similarity_map(gradient_image, gradient_reference, self.C1)
        gradient_similarity_RF = similarity_map(gradient_reference, gradient_fusion, self.C2)
        gradient_similarity_DF = similarity_map(gradient_image, gradient_fusion, self.C2)
        gradient_similarity = gradient_similarity_RD + gradient_similarity_DF - gradient_similarity_RF

        # chromaticity similarity
        chromaticity_similarity = (
            2 * (image_lhm[1] * reference_lhm[1] + image_lhm[2] * reference_lhm[2]) + self.C3
        ) / (image_lhm[1] ** 2 + reference_lhm[1] ** 2 + image_lhm[2] ** 2 + reference_lhm[2] ** 2 + self.C3)

        # convert to complex numbers to handle negative numberes and exponents smaller than 1 without NaNs
        gradient_similarity = gradient_similarity.to(torch.complex128)
        chromaticity_similarity = chromaticity_similarity.to(torch.complex128)

        # gradient-chromaticity similarity
        if self.combination_method == "sum":
            gradient_chromaticity_similarity = (
                self.alpha * gradient_similarity + (1 - self.alpha) * chromaticity_similarity
            )
        elif self.combination_method == "product":
            gradient_chromaticity_similarity = gradient_similarity**self.gamma * chromaticity_similarity**self.beta

        # deviation pooling
        mean_gradient_chromaticity_similarity = torch.mean(gradient_chromaticity_similarity**self.q)
        return (
            torch.mean(
                torch.abs(gradient_chromaticity_similarity**self.q - mean_gradient_chromaticity_similarity) ** self.rho
            )
            ** self.o
        )

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(
            implementations,
            "piq",
            get_piq_mdsi(
                combination=self.combination_method,
                c1=self.C1,
                c2=self.C2,
                c3=self.C3,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                q=self.q,
                rho=self.rho,
                o=self.o,
            ),
        )
        self._register_implementation(implementations, "piqa", get_piqa_mdsi(self.combination_method))
        # piqa's MDSI implementation only allows to set combination method,
        # other parameters are fixed to default values of original paper

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) "
            f"{self._arrow_indicating_optimum()} "
            f"with combination_method={self.combination_method}, "
            f"C1={self.C1}, C2={self.C2}, C3={self.C3}, "
            f"alpha={self.alpha}, gamma={self.gamma}, beta={self.beta}, q={self.q}, rho={self.rho}, o={self.o}"
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
        check_value_range(image, 0, 255)
        check_value_range(reference, 0, 255, reference=True)

        warn_if_all_pixels_in_0_to_1_range(image, reference)

        check_rgb(self.expected_dimensions, image, self.abbreviation)
