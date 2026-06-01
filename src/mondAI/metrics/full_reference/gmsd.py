from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.piq import get_piq_gmsd
from mondAI.metrics.third_party.piqa import get_piqa_gmsd
from mondAI.utils.signal_processing import gradient_map, subsample
from mondAI.utils.similarity_map import similarity_map

logger = get_logger()


class GMSD(FullReferenceMetric):
    """Gradient Magnitude Similarity Deviation (GMSD)

    This metric computes the standard deviation of the gradient magnitude similarity map between a distorted image
    and a reference image. It expects grayscale images with pixel values in the range [0, 255] and yields scores
    between 0 and infinity, where lower values indicate better perceptual quality. GMSD is designed to be a simple
    and efficient metric that captures local variations in image quality based on gradient information.

    Implementation adapted from original MATLAB implementation by Wufeng Xue, Lei Zhang, Xuanqin Mou, and Alan C. Bovik,
    which is available at: http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm

    Original publication:
    Xue, Wufeng, et al.
    "Gradient magnitude similarity deviation: A highly efficient perceptual image quality index."
    IEEE transactions on image processing 23.2 (2013): 684-695.

    """

    @property
    def name(self) -> str:
        return "Gradient Magnitude Similarity Deviation"

    @property
    def abbreviation(self) -> str:
        return "GMSD"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, t: float = 170.0) -> None:
        """
        :param t: Constant to avoid numerical instability. Default is 170.0,
        which is related to c = 0.0026 = t / (255.0^2) as emperically defined in the original paper.
        :type t: float
        """
        super().__init__()
        self.t = t

        if self.t != 170.0:
            logger.warning(
                f"Using a non-default value of t={self.t} for GMSD may lead to results "
                "that are not directly comparable to the original formulation of GMSD, "
                "which uses t=170.0. Please ensure that you understand the implications "
                "of changing this parameter on the metric's behavior and interpretability."
            )

        if self.t < 0:
            raise ValueError(f"t must be non-negative, but got {self.t}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self._input_checks(image, reference)

        # subsample by 2
        image = subsample(image)
        reference = subsample(reference)

        # Compute gradient magnitude maps
        prewitt_kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], device=image.device, dtype=image.dtype) / 3
        gradient_map_image = gradient_map(image, prewitt_kernel)
        gradient_map_reference = gradient_map(reference, prewitt_kernel)

        gradient_similarity = similarity_map(gradient_map_image, gradient_map_reference, self.t)

        return torch.std(gradient_similarity)

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        # as piq and piqa expect 0-1 range and c is defined as 170.0/255.0^2 in the original paper, we need to adjust t
        self._register_implementation(implementations, "piq", get_piq_gmsd(self.t / (255.0**2)))
        self._register_implementation(implementations, "piqa", get_piqa_gmsd(self.t / (255.0**2)))

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} with c={self.t}"

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to GMSD, such as checking for valid pixel
        value ranges and dimensions.

        Warns if all pixel values in both image and reference are in the range [0, 1],
        which may indicate that the images are not correctly scaled for GMSD, which
        expects pixel values in the range [0, 255].

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :raises ValueError: If the input image contains pixel values outside the range
            [0, 255].
        :raises ValueError: If the reference image contains pixel values outside the
            range [0, 255].

        """
        # Input checks specific to GMSD
        if torch.any(image < 0) or torch.any(image > 255):
            raise ValueError("Input image contains pixel values outside the range [0, 255].")

        if torch.any(reference < 0) or torch.any(reference > 255):
            raise ValueError("Reference image contains pixel values outside the range [0, 255].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            logger.warning(
                "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "GMSD expects pixel values in the range [0, 255]. Please ensure that your input images are "
                "correctly scaled for accurate computation of GMSD."
            )
