import math
from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.piq import get_piq_dss
from mondAI.utils.signal_processing import convolve2d, gaussian_filter_kernel

logger = get_logger()


class DSS(FullReferenceMetric):
    r"""DCT subband Similarity (DSS) is a full reference image quality assessment metric
    that evaluates the similarity between two images based on their discrete cosine
    transform (DCT) subband representations. The metric computes the similarity of the
    DCT coefficients in different subbands, which can capture both spatial and
    frequency information of the images. DSS is designed to be sensitive to distortions
    that affect the perceptual quality of images, making it a useful tool for
    evaluating image processing algorithms and comparing the quality of different
    images.

    DSS expects grayscale images with pixel values in the range [0, 255].
    It is symmetric, meaning that the order of the input images does not affect the result.
    DSS yields scores in the range [0, 1], where higher values indicate better perceptual quality.

    It is defined as

    ..math::
        \operatorname{DSS} = \sum_{m=0}^{7} \sum_{n=0}^{7} w_{m,n} \cdot \operatorname{DSS_{m,n}}

    where :math:`w_{m,n}` is a Gaussian weight for the subband and :math:`\operatorname{DSS_{m,n}}` is the similarity
    score for the subband computed based on the local variance of the DCT coefficients in that subband by:

    ..math::
    \operatorname{DSS_{m,n}(p, q)} =
        \frac{2 {\sigma_{m,n}^X (p,q) \sigma_{m,n}^Y (p,q)} + C}{\sigma_{m,n}^X (p,q)^2 + \sigma_{m,n}^Y (p,q)^2 + C}

    where :math:`\sigma_{m,n}^X (p,q)` and :math:`\sigma_{m,n}^Y (p,q)` are the local variances at location (p,q) of
    the DCT coefficients in the subband for the image and reference, respectively. And :math:`C` are constants for
    numerical stability, with a different value for the DC subband and the other subbands.


    The DC subband (m=0, n=0) is treated differently from the other subbands, as it captures the average intensity of
    the image and is more sensitive to certain types of distortions. It is defined as:

    ..math::
    \operatorname{DSS_{0,0}(p, q)} =
        \frac{2 {\sigma_{0,0}^X (p,q) \sigma_{0,0}^Y (p,q)} + C}{\sigma_{0,0}^X (p,q)^2 + \sigma_{0,0}^Y (p,q)^2 + C}
        \cdot \frac{\sigma_{0,0}^{XY}(p,q) + C}{\sigma_{0,0}^X (p,q) \sigma_{0,0}^Y (p,q) + C}


    Implementation adapted from original MATLAB implementation by Yair Moshe:
    https://de.mathworks.com/matlabcentral/fileexchange/53708-dct-subband-similarity-index-for-measuring-image-quality
    licence: BSD 3-Clause License

    Original publication:
    Balanov, A., Schwartz, A., Moshe, Y., & Peleg, N. (2015, September).
    Image quality assessment based on DCT subband similarity.
    In 2015 IEEE international conference on image processing (ICIP) (pp. 2105-2109). IEEE.

    """

    @property
    def name(self) -> str:
        return "DCT Subband Similarity"

    @property
    def abbreviation(self) -> str:
        return "DSS"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, sigma: float = 1.55, C: tuple[float, float] = (1000.0, 300.0)) -> None:
        """Initialize the Metric Template.

        :param sigma: The standard deviation of the Gaussian filter used in the metric,
            must be positive, default is 1.55.
        :type sigma: float
        :param C: A tuple of two numerical stability constants used in the metric, must
            be positive, default is (1000.0, 300.0).
        :type C: tuple[float, float]
        :raises ValueError: If sigma is not positive or if any value in C is not
            positive.

        """
        super().__init__()
        self.sigma = sigma
        self.C = C

        if self.sigma <= 0:
            raise ValueError(f"Sigma must be positive, but got {self.sigma}.")

        if any(c <= 0 for c in self.C):
            raise ValueError(f"C must be positive, but got {self.C}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self._input_checks(image, reference)

        # crop images to the closest multiple of 8 in height and width
        height, width = image.shape[-2:]
        new_height = (height // 8) * 8
        new_width = (width // 8) * 8
        image = image[:new_height, :new_width]
        reference = reference[:new_height, :new_width]

        # channel decpmposition by 8x8 2D DCT
        image_decomposed = self._dct_decompose(image)
        reference_decomposed = self._dct_decompose(reference)

        # compute similarity of DCT coefficients in different subbands
        subband_similarities = image.new_zeros(8, 8)
        small_weight_threshold = 1e-2

        # compute weights for each subband
        u = torch.arange(8, device=image.device, dtype=image.dtype)
        X, Y = torch.meshgrid(u, u, indexing="ij")
        distances = torch.sqrt((X + 0.5) ** 2 + (Y + 0.5) ** 2)
        w = torch.exp(-(distances**2) / (2 * self.sigma**2))

        for m in range(8):
            for n in range(8):
                # skip subbands with very small weight
                if w[m, n] < small_weight_threshold:
                    w[m, n] = 0.0
                    continue

                dc = m == 0 and n == 0
                subband_similarities[m, n] = self._compute_subband_similarity(
                    image_decomposed[m::8, n::8], reference_decomposed[m::8, n::8], dc
                )

        return torch.sum(subband_similarities * (w / torch.sum(w)))

    def _dct_decompose(self, image: torch.Tensor) -> torch.Tensor:
        """Decompose the image into 8x8 blocks and apply 2D DCT to each block.

        :param image: The input image to be decomposed, shape (H, W)
        :type image: torch.Tensor
        :return: The DCT coefficients of the image, shape (H, W)
        :rtype: torch.Tensor

        """
        D = self._dctmtx(8, device=image.device, dtype=image.dtype)
        decomposed = image.new_zeros(image.shape)

        for block_num_in_row in range(image.shape[0] // 8):
            for block_num_in_col in range(image.shape[1] // 8):
                block = image[
                    block_num_in_row * 8 : (block_num_in_row + 1) * 8, block_num_in_col * 8 : (block_num_in_col + 1) * 8
                ]
                decomposed[
                    block_num_in_row * 8 : (block_num_in_row + 1) * 8, block_num_in_col * 8 : (block_num_in_col + 1) * 8
                ] = D @ block @ D.T

        return decomposed

    def _dctmtx(self, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate the DCT matrix of size NxN.

        :param N: The size of the DCT matrix (must be positive).
        :type N: int
        :param device: The device on which to create the DCT matrix.
        :type device: torch.device
        :param dtype: The data type of the DCT matrix.
        :type dtype: torch.dtype
        :return: The DCT matrix of size NxN.
        :rtype: torch.Tensor

        """
        if N <= 0:
            raise ValueError(f"N must be positive, but got {N}.")

        x = torch.arange(N, device=device, dtype=dtype)
        k = x.unsqueeze(1)
        n = x.unsqueeze(0)
        dct_matrix = torch.cos(math.pi / N * (n + 0.5) * k)
        dct_matrix[0] *= 1 / math.sqrt(N)
        dct_matrix[1:] *= math.sqrt(2 / N)
        return dct_matrix

    def _compute_subband_similarity(
        self, image_subband: torch.Tensor, reference_subband: torch.Tensor, dc: bool
    ) -> torch.Tensor:
        """Compute the similarity of DCT coefficients in a specific subband.

        :param image_subband: The DCT coefficients of the image in the subband, shape
            (H/8, W/8)
        :type image_subband: torch.Tensor
        :param reference_subband: The DCT coefficients of the reference image in the
            subband, shape (H/8, W/8)
        :type reference_subband: torch.Tensor
        :param dc: A boolean indicating whether the subband is the DC subband.
        :type dc: bool
        :return: The similarity score for the subband.
        :rtype: torch.Tensor

        """
        c = self.C[0] if dc else self.C[1]

        # compute local variance
        window = gaussian_filter_kernel(3, 1.5, device=image_subband.device, dtype=image_subband.dtype)
        mu_image = convolve2d(image_subband, window, padding="same")
        mu_reference = convolve2d(reference_subband, window, padding="same")
        sigma_image_squared = convolve2d(image_subband**2, window, padding="same") - mu_image**2
        sigma_reference_squared = convolve2d(reference_subband**2, window, padding="same") - mu_reference**2

        sigma_image_squared[sigma_image_squared < 0] = 0.0
        sigma_reference_squared[sigma_reference_squared < 0] = 0.0

        variance_left_term = (2 * torch.sqrt(sigma_image_squared * sigma_reference_squared) + c) / (
            sigma_image_squared + sigma_reference_squared + c
        )

        # spatial pooling of worst scores
        percentile = 0.05
        percentile_index = round(percentile * variance_left_term.numel())
        sorted_variance_left_term = torch.sort(variance_left_term.flatten()).values
        similarity = sorted_variance_left_term[:percentile_index].mean()

        if dc:
            # For DC, multiply by a right term
            sigma_cross = (
                convolve2d(image_subband * reference_subband, window, padding="same") - mu_image * mu_reference
            )
            variance_right_term = (sigma_cross + c) / (torch.sqrt(sigma_image_squared * sigma_reference_squared) + c)
            percentile_index = round(percentile * variance_right_term.numel())
            sorted_variance_right_term = torch.sort(variance_right_term.flatten()).values
            similarity *= sorted_variance_right_term[:percentile_index].mean()

        return similarity

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        """Override this method in subclasses to register other implementations of the
        metric for comparison. Use the `_register_implementation` helper method to add
        implementations to the internal dictionary. These implementations will be used
        when compare_implementations is True.

        Store reference implementations in the `third_party` submodule of the metrics module,
        and import them here to register them for comparison.

        """
        self._register_implementation(implementations, "piq", get_piq_dss(self.sigma))

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} with sigma={self.sigma}, C={self.C}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to DSS, such as checking for valid pixel value
        ranges and dimensions.

        Warns if all pixel values in both image and reference are in the range [0, 1],
        which may indicate that the images are not correctly scaled for DSS, which
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
        # Input checks specific to DSS
        if torch.any(image < 0) or torch.any(image > 255):
            raise ValueError("Input image contains pixel values outside the range [0, 255].")

        if torch.any(reference < 0) or torch.any(reference > 255):
            raise ValueError("Reference image contains pixel values outside the range [0, 255].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            logger.warning(
                "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "DSS expects pixel values in the range [0, 255]. Please ensure that your input images are "
                "correctly scaled for accurate computation of DSS."
            )
