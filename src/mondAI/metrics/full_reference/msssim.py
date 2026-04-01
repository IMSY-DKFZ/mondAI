from typing import Callable, Sequence

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.utils.signal_processing import convolve2d, gaussian_filter_kernel

logger = get_logger()


# TODO: Revisit docstrings
class MSSSIM(FullReferenceMetric):
    r"""Multi-Scale Structural Similarity Index Measure (MS-SSIM).

    The Multi-Scale Structural Similarity Index Measure (MS-SSIM) extends SSIM by
    evaluating image similarity across multiple scales. At each scale, the images are
    compared using local luminance, contrast, and structure statistics, and then
    downsampled for the next coarser scale. The final score combines the contrast-
    structure terms from the coarser levels with the full SSIM score from the final
    level.

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

    - this implementation is grayscale-only for now,
    - it uses valid convolution for the per-scale SSIM computation, matching the
      original MATLAB implementation,
    - it uses symmetric-padding downsampling with a 2x2 averaging filter between
      scales,
    - it supports both the original weighted ``product`` aggregation and the
      alternative ``wtd_sum`` mode from the MATLAB reference code,
    - it expects images in ``[0, dynamic_range]`` and defaults to ``dynamic_range=255``.

    """

    DEFAULT_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    @property
    def name(self) -> str:
        return "Multi-Scale Structural Similarity Index Measure"

    @property
    def abbreviation(self) -> str:
        return "MSSSIM"

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
        levels: int = 5,
        weights: Sequence[float] | None = None,
        method: str = "product",
    ) -> None:
        """Initialize MS-SSIM with defaults matching the original implementation."""
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.dynamic_range = dynamic_range
        self.levels = levels
        self.weights = tuple(weights) if weights is not None else self.DEFAULT_WEIGHTS
        self.method = method

        if self.k1 < 0 or self.k2 < 0:
            raise ValueError("k1 and k2 must be non-negative.")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if self.kernel_size * self.kernel_size < 4:
            raise ValueError("kernel_size must define a window with at least 4 elements.")
        if self.kernel_sigma <= 0:
            raise ValueError("kernel_sigma must be positive.")
        if self.dynamic_range <= 0:
            raise ValueError("dynamic_range must be positive.")
        if self.levels < 1:
            raise ValueError("levels must be at least 1.")
        if len(self.weights) != self.levels:
            raise ValueError("weights must have the same length as levels.")
        if sum(self.weights) == 0:
            raise ValueError("weights must not sum to zero.")
        if self.method not in ("product", "wtd_sum"):
            raise ValueError("method must be either 'product' or 'wtd_sum'.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute MS-SSIM between image and reference."""
        self._input_checks(image, reference)

        image = image.to(torch.float64)
        reference = reference.to(torch.float64)

        mssim_values: list[torch.Tensor] = []
        mcs_values: list[torch.Tensor] = []

        downsample_filter = torch.ones((2, 2), device=image.device, dtype=image.dtype) / 4.0

        for _ in range(self.levels):
            mssim, mcs = self._compute_ssim_and_cs(image, reference)
            mssim_values.append(mssim)
            mcs_values.append(mcs)

            filtered_image = convolve2d(image, downsample_filter, padding="same")
            filtered_reference = convolve2d(reference, downsample_filter, padding="same")

            image = filtered_image[::2, ::2]
            reference = filtered_reference[::2, ::2]

        weights = torch.tensor(self.weights, device=image.device, dtype=image.dtype)
        mssim_tensor = torch.stack(mssim_values)
        mcs_tensor = torch.stack(mcs_values)

        if self.method == "product":
            if self.levels == 1:
                result = mssim_tensor[0] ** weights[0]
            else:
                result = torch.prod(mcs_tensor[:-1] ** weights[:-1]) * (mssim_tensor[-1] ** weights[-1])
        else:
            normalized_weights = weights / weights.sum()
            if self.levels == 1:
                result = mssim_tensor[0]
            else:
                result = (
                    torch.sum(mcs_tensor[:-1] * normalized_weights[:-1]) + mssim_tensor[-1] * normalized_weights[-1]
                )

        return result.to(dtype=torch.float64)

    def _compute_ssim_and_cs(self, image: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean SSIM and contrast-structure terms for one scale."""
        kernel = gaussian_filter_kernel(self.kernel_size, self.kernel_sigma, device=image.device, dtype=image.dtype)

        c1 = (self.k1 * self.dynamic_range) ** 2
        c2 = (self.k2 * self.dynamic_range) ** 2

        mu_image = convolve2d(image, kernel, padding="valid")
        mu_reference = convolve2d(reference, kernel, padding="valid")

        mu_image_sq = mu_image * mu_image
        mu_reference_sq = mu_reference * mu_reference
        mu_image_reference = mu_image * mu_reference

        sigma_image_sq = convolve2d(image * image, kernel, padding="valid") - mu_image_sq
        sigma_reference_sq = convolve2d(reference * reference, kernel, padding="valid") - mu_reference_sq
        sigma_image_reference = convolve2d(image * reference, kernel, padding="valid") - mu_image_reference

        if c1 > 0 and c2 > 0:
            cs_map = (2 * sigma_image_reference + c2) / (sigma_image_sq + sigma_reference_sq + c2)
            ssim_map = (
                (2 * mu_image_reference + c1)
                * (2 * sigma_image_reference + c2)
                / ((mu_image_sq + mu_reference_sq + c1) * (sigma_image_sq + sigma_reference_sq + c2))
            )
        else:
            numerator1 = 2 * mu_image_reference + c1
            numerator2 = 2 * sigma_image_reference + c2
            denominator1 = mu_image_sq + mu_reference_sq + c1
            denominator2 = sigma_image_sq + sigma_reference_sq + c2

            cs_map = torch.ones_like(mu_image)
            ssim_map = torch.ones_like(mu_image)

            valid_cs = denominator2 > 0
            cs_map[valid_cs] = numerator2[valid_cs] / denominator2[valid_cs]

            valid_ssim = denominator1 * denominator2 > 0
            ssim_map[valid_ssim] = (
                numerator1[valid_ssim] * numerator2[valid_ssim] / (denominator1[valid_ssim] * denominator2[valid_ssim])
            )

            fallback_ssim = (denominator1 != 0) & (denominator2 == 0)
            ssim_map[fallback_ssim] = numerator1[fallback_ssim] / denominator1[fallback_ssim]

        return ssim_map.mean(), cs_map.mean()

    # TODO: Revisit comments on other implementations
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
                ).to(image.dtype)

            implementations["torchmetrics"] = torchmetrics_msssim
        except Exception:
            logger.warning(
                "torchmetrics or its MS-SSIM implementation is not available, "
                "skipping torchmetrics implementation of MSSSIM"
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
                return torch.tensor(score.item(), device=image.device, dtype=image.dtype)

            implementations["tensorflow"] = tensorflow_msssim
        except Exception:
            logger.warning("tensorflow is not available, skipping tensorflow implementation of MSSSIM")

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
                ).to(image.dtype)

            implementations["piq"] = piq_msssim
        except Exception:
            logger.warning("piq or its MS-SSIM implementation is not available, skipping piq implementation of MSSSIM")

        try:
            from piqa import MS_SSIM

            metric = MS_SSIM(
                window_size=self.kernel_size,
                sigma=self.kernel_sigma,
                n_channels=1,
                reduction="mean",
                value_range=self.dynamic_range,
                weights=self.weights,
                k1=self.k1,
                k2=self.k2,
            )

            def piqa_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # PIQA exposes MS-SSIM as a module and expects NCHW tensors.
                return metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0)).to(image.dtype)

            implementations["piqa"] = piqa_msssim
        except Exception:
            logger.warning(
                "piqa or its MS-SSIM implementation is not available, skipping piqa implementation of MSSSIM"
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
        except Exception:
            logger.warning(
                "sewar or its MS-SSIM implementation is not available, skipping sewar implementation of MSSSIM"
            )

        try:
            from monai.metrics import MultiScaleSSIMMetric

            metric = MultiScaleSSIMMetric(
                spatial_dims=2,
                data_range=self.dynamic_range,
                kernel_type="gaussian",
                win_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
                weights=self.weights,
                k1=self.k1,
                k2=self.k2,
            )

            def monai_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # MONAI exposes MS-SSIM as a metric object for batched medical imaging.
                return metric(reference.unsqueeze(0).unsqueeze(0), image.unsqueeze(0).unsqueeze(0)).to(image.dtype)

            implementations["monai"] = monai_msssim
        except Exception:
            logger.warning(
                "monai or its MS-SSIM implementation is not available, skipping monai implementation of MSSSIM"
            )

        try:
            from medimetrics.metrics import SSIM as MediMetricsSSIM

            metric = MediMetricsSSIM()

            def medimetrics_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # medimetrics exposes MS-SSIM through the SSIM metric class using a
                # NumPy-based compute method for medical image quality assessment.
                score = metric.compute(reference.cpu().numpy(), image.cpu().numpy(), multi_scale=True)
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["medimetrics"] = medimetrics_msssim
        except Exception:
            logger.warning(
                "medimetrics or its MS-SSIM implementation is not available, "
                "skipping medimetrics implementation of MSSSIM"
            )

        return implementations

    def __str__(self) -> str:
        """Full text representation of the metric."""
        arrow = self._arrow_indicating_optimum()
        return (
            f"{self.name} ({self.abbreviation}) {arrow} with parameters: "
            f"{self.k1=}, {self.k2=}, {self.kernel_size=}, {self.kernel_sigma=}, "
            f"{self.dynamic_range=}, {self.levels=}, {self.weights=}, {self.method=}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to MS-SSIM."""
        if torch.any(image < 0) or torch.any(image > self.dynamic_range):
            raise ValueError(f"Input image contains pixel values outside the range [0, {self.dynamic_range}].")

        if torch.any(reference < 0) or torch.any(reference > self.dynamic_range):
            raise ValueError(f"Reference image contains pixel values outside the range [0, {self.dynamic_range}].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            if self.dynamic_range != 1.0:
                logger.warning(
                    "It has been detected that all pixel values in both image and reference are "
                    "in the range [0, 1]. MSSSIM defaults to dynamic_range=255. Please ensure that "
                    "your input images are correctly scaled or set dynamic_range=1.0 for normalized inputs."
                )

        if image.shape[-2] < self.kernel_size or image.shape[-1] < self.kernel_size:
            raise ValueError(
                f"Images have spatial dimensions {image.shape[-2:]} which are smaller than the required window size "
                f"{self.kernel_size}x{self.kernel_size} for MSSSIM."
            )

        minimum_image_width = min(image.shape[-2], image.shape[-1]) / (2 ** (self.levels - 1))
        if minimum_image_width < self.kernel_size:
            raise ValueError(
                "Images become too small for the requested number of MS-SSIM levels and window size after repeated "
                "downsampling."
            )
