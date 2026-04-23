from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.utils.signal_processing import convolve2d
from mondAI.utils.similarity_map import ssim_and_cs_maps

logger = get_logger()


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
    "Multi-scale structural similarity for image quality assessment,"
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

        :param k1: First stability constant coefficient, default is 0.01.
        :type k1: float
        :param k2: Second stability constant coefficient, default is 0.03.
        :type k2: float
        :param kernel_size: Size of the Gaussian window, default is 11.
        :type kernel_size: int
        :param kernel_sigma: Standard deviation of the Gaussian window, default is 1.5.
        :type kernel_sigma: float
        :param dynamic_range: Dynamic range ``L`` of the images, default is 255.0.
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

        downsample_filter = image.new_ones((2, 2)) / 4.0

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

            filtered_image = convolve2d(image, downsample_filter, padding="same")
            filtered_reference = convolve2d(reference, downsample_filter, padding="same")

            image = filtered_image[::2, ::2]
            reference = filtered_reference[::2, ::2]

        weights = torch.tensor(self.weights, device=image.device, dtype=image.dtype)
        terms = torch.cat((mean_contrasts_and_structures[:-1], mean_ssims[-1:]))

        if self.method == "product":
            return torch.prod(terms**weights)

        elif self.method == "weighted sum":
            normalized_weights = weights / weights.sum()
            return torch.sum(terms * normalized_weights)

        else:
            return None  # This should never happen due to the check in __init__

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return other MS-SSIM implementations for comparison."""
        implementations = {}

        try:
            from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

            def torchmetrics_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # torchmetrics expects NCHW tensors and implements the multiscale
                # aggregation internally.
                return multiscale_structural_similarity_index_measure(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    gaussian_kernel=True,
                    sigma=self.kernel_sigma,
                    kernel_size=self.kernel_size,
                    reduction="elementwise_mean",
                    data_range=self.dynamic_range,
                    k1=self.k1,
                    k2=self.k2,
                    betas=self.weights,
                    normalize=None,
                )

            implementations["torchmetrics"] = torchmetrics_msssim
        except ImportError:
            logger.warning(
                "torchmetrics or its MS-SSIM implementation is not available, "
                "skipping torchmetrics implementation of MS-SSIM"
            )

        try:
            import tensorflow as tf

            def tensorflow_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # TensorFlow expects NHWC tensors and supports configurable scale weights.
                score = tf.image.ssim_multiscale(
                    image.cpu().numpy()[None, ..., None],
                    reference.cpu().numpy()[None, ..., None],
                    max_val=self.dynamic_range,
                    power_factors=self.weights,
                    filter_size=self.kernel_size,
                    filter_sigma=self.kernel_sigma,
                    k1=self.k1,
                    k2=self.k2,
                ).numpy()
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["tensorflow"] = tensorflow_msssim
        except ImportError:
            logger.warning("tensorflow is not available, skipping tensorflow implementation of MS-SSIM")

        try:
            from piq import multi_scale_ssim

            def piq_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # PIQ expects NCHW tensors and exposes the number of scales through the weights.
                return multi_scale_ssim(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    kernel_size=self.kernel_size,
                    kernel_sigma=self.kernel_sigma,
                    data_range=self.dynamic_range,
                    reduction="mean",
                    scale_weights=torch.tensor(self.weights, device=image.device, dtype=image.dtype),
                    k1=self.k1,
                    k2=self.k2,
                )

            implementations["piq"] = piq_msssim
        except ImportError:
            logger.warning("piq or its MS-SSIM implementation is not available, skipping piq implementation of MS-SSIM")

        try:
            from piqa import MS_SSIM

            def piqa_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # PIQA exposes MS-SSIM as a module and expects NCHW tensors.
                piqa_metric = MS_SSIM(
                    window_size=self.kernel_size,
                    sigma=self.kernel_sigma,
                    n_channels=1,
                    reduction="mean",
                    value_range=self.dynamic_range,
                    weights=torch.tensor(self.weights, device=image.device, dtype=image.dtype),
                    k1=self.k1,
                    k2=self.k2,
                ).to(image.device)

                return piqa_metric(image.float().unsqueeze(0).unsqueeze(0), reference.float().unsqueeze(0).unsqueeze(0))

            implementations["piqa"] = piqa_msssim
        except ImportError:
            logger.warning(
                "piqa or its MS-SSIM implementation is not available, skipping piqa implementation of MS-SSIM"
            )

        try:
            from sewar.full_ref import msssim as msssim_sewar

            def sewar_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # sewar operates on NumPy arrays and follows a classic image-quality API.
                score = msssim_sewar(
                    reference.cpu().numpy(),
                    image.cpu().numpy(),
                    weights=self.weights,
                    ws=self.kernel_size,
                    K1=self.k1,
                    K2=self.k2,
                    MAX=self.dynamic_range,
                )
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["sewar"] = sewar_msssim
        except ImportError:
            logger.warning(
                "sewar or its MS-SSIM implementation is not available, skipping sewar implementation of MS-SSIM"
            )

        try:
            from monai.metrics import MultiScaleSSIMMetric

            monai_metric = MultiScaleSSIMMetric(
                spatial_dims=2,
                data_range=self.dynamic_range,
                kernel_type="gaussian",
                kernel_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
                weights=self.weights,
                k1=self.k1,
                k2=self.k2,
            )

            def monai_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # MONAI exposes MS-SSIM as a metric object for batched medical imaging.
                return monai_metric(reference.unsqueeze(0).unsqueeze(0), image.unsqueeze(0).unsqueeze(0)).to(
                    image.dtype
                )

            implementations["monai"] = monai_msssim
        except ImportError:
            logger.warning(
                "monai or its MS-SSIM implementation is not available, skipping monai implementation of MS-SSIM"
            )

        try:
            from mondAI.metrics.third_party.medimetrics.ssim import MSSSIM as MediMetricsMSSSIM

            metric = MediMetricsMSSSIM()

            def medimetrics_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # medimetrics exposes MS-SSIM through the SSIM metric class using a
                # NumPy-based compute method for medical image quality assessment.
                score = metric.compute(reference.cpu().numpy(), image.cpu().numpy(), multi_scale=True)
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["medimetrics"] = medimetrics_msssim
        except ImportError:
            logger.warning(
                "medimetrics or its MS-SSIM implementation is not available, "
                "skipping medimetrics implementation of MS-SSIM"
            )

        return implementations

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
                "It has been detected that all pixel values in both image and reference are "
                "in the range [0, 1]. MS-SSIM defaults to dynamic_range=255. Please ensure that "
                "your input images are correctly scaled or set dynamic_range=1.0 for normalized inputs."
            )

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
