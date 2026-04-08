from typing import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.utils.signal_processing import convolve2d, gaussian_filter_kernel

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
    ) -> None:
        """Initialize SSIM with defaults matching the original implementation.

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
        :raises ValueError: If parameters are invalid.

        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.dynamic_range = dynamic_range

        if self.k1 < 0 or self.k2 < 0:
            raise ValueError("k1 and k2 must be non-negative.")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}.")
        if self.kernel_size * self.kernel_size < 4:
            raise ValueError("kernel_size must define a window with at least 4 elements.")
        if self.kernel_sigma <= 0:
            raise ValueError("kernel_sigma must be positive.")
        if self.dynamic_range <= 0:
            raise ValueError("dynamic_range must be positive.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between image and reference."""
        self._input_checks(image, reference)

        kernel = gaussian_filter_kernel(self.kernel_size, self.kernel_sigma, device=image.device, dtype=image.dtype)

        c1 = (self.k1 * self.dynamic_range) ** 2
        c2 = (self.k2 * self.dynamic_range) ** 2

        mu_image = convolve2d(image, kernel, padding="valid")
        mu_reference = convolve2d(reference, kernel, padding="valid")

        mu_image_squared = mu_image * mu_image
        mu_reference_squared = mu_reference * mu_reference
        mu_image_reference = mu_image * mu_reference

        sigma_image_squared = convolve2d(image * image, kernel, padding="valid") - mu_image_squared
        sigma_reference_squared = convolve2d(reference * reference, kernel, padding="valid") - mu_reference_squared
        sigma_image_reference = convolve2d(image * reference, kernel, padding="valid") - mu_image_reference

        if c1 > 0 and c2 > 0:
            ssim_map = ((2 * mu_image_reference + c1) * (2 * sigma_image_reference + c2)) / (
                (mu_image_squared + mu_reference_squared + c1) * (sigma_image_squared + sigma_reference_squared + c2)
            )
        else:
            numerator1 = 2 * mu_image_reference + c1
            numerator2 = 2 * sigma_image_reference + c2
            denominator1 = mu_image_squared + mu_reference_squared + c1
            denominator2 = sigma_image_squared + sigma_reference_squared + c2

            ssim_map = torch.ones_like(mu_image)
            valid_index = denominator1 * denominator2 > 0
            ssim_map[valid_index] = (
                numerator1[valid_index]
                * numerator2[valid_index]
                / (denominator1[valid_index] * denominator2[valid_index])
            )

            fallback_index = (denominator1 != 0) & (denominator2 == 0)
            ssim_map[fallback_index] = numerator1[fallback_index] / denominator1[fallback_index]

        return ssim_map.mean()  # TODO: Also expose the raw SSIM map (and possibly gradients; cf. scikit-image).

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return other SSIM implementations for comparison."""
        implementations = {}

        try:
            from skimage.metrics import structural_similarity

            def skimage_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # scikit-image is the closest direct reference implementation.
                # Differences relative to mondAI are usually limited to floating-point
                # precision, especially when operating on float32 inputs.
                score = structural_similarity(
                    reference.cpu().numpy(),
                    image.cpu().numpy(),
                    win_size=self.kernel_size,
                    gradient=False,
                    data_range=self.dynamic_range,
                    channel_axis=None,
                    gaussian_weights=True,
                    full=False,
                    use_sample_covariance=False,
                    K1=self.k1,
                    K2=self.k2,
                    sigma=self.kernel_sigma,
                )
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["scikit-image"] = skimage_ssim
        except Exception:
            logger.warning(
                "scikit-image or its SSIM implementation is not available, skipping scikit-image implementation of SSIM"
            )

        try:
            from torchmetrics.functional.image import structural_similarity_index_measure

            def torchmetrics_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # torchmetrics expects NCHW tensors. Its implementation differs from
                # the original MATLAB code, for example by using reflection padding
                # instead of strict valid convolution.
                return structural_similarity_index_measure(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    gaussian_kernel=True,
                    sigma=self.kernel_sigma,
                    kernel_size=self.kernel_size,
                    reduction="elementwise_mean",
                    data_range=self.dynamic_range,
                    k1=self.k1,
                    k2=self.k2,
                    return_full_image=False,
                    return_contrast_sensitivity=False,
                )

            implementations["torchmetrics"] = torchmetrics_ssim
        except Exception:
            logger.warning(
                "torchmetrics or its SSIM implementation is not available, skipping torchmetrics implementation of SSIM"
            )

        try:
            import tensorflow as tf

            def tensorflow_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # TensorFlow expects NHWC tensors. Small deviations may arise from
                # its internal use of single precision (float32) and its own
                # implementation of the Gaussian filter construction.
                score = tf.image.ssim(
                    image.cpu().numpy()[None, ..., None],
                    reference.cpu().numpy()[None, ..., None],
                    max_val=self.dynamic_range,
                    filter_size=self.kernel_size,
                    filter_sigma=self.kernel_sigma,
                    k1=self.k1,
                    k2=self.k2,
                ).numpy()
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["tensorflow"] = tensorflow_ssim
        except Exception:
            logger.warning("tensorflow is not available, skipping tensorflow implementation of SSIM")

        try:
            from piq import ssim

            def piq_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # PIQ expects NCHW tensors. In practice, its results are typically
                # very close to scikit-image, with only minor differences stemming
                # from implementation details such as kernel construction.
                return ssim(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    kernel_size=self.kernel_size,
                    kernel_sigma=self.kernel_sigma,
                    data_range=self.dynamic_range,
                    reduction="mean",
                    full=False,
                    downsample=False,
                    k1=self.k1,
                    k2=self.k2,
                )

            implementations["piq"] = piq_ssim
        except Exception:
            logger.warning("piq or its SSIM implementation is not available, skipping piq implementation of SSIM")

        try:
            from piqa import SSIM as PIQASSIM

            piqa_metric = PIQASSIM(
                window_size=self.kernel_size,
                sigma=self.kernel_sigma,
                n_channels=1,
                reduction="mean",
                value_range=self.dynamic_range,
                k1=self.k1,
                k2=self.k2,
            )

            def piqa_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # PIQA exposes SSIM as a module. It expects NCHW tensors and an
                # explicit channel count. In practice, it behaves very similarly to
                # scikit-image for the parameter settings used here.
                piqa_metric.to(image.device)
                return piqa_metric(image.float().unsqueeze(0).unsqueeze(0), reference.float().unsqueeze(0).unsqueeze(0))

            implementations["piqa"] = piqa_ssim
        except Exception:
            logger.warning("piqa or its SSIM implementation is not available, skipping piqa implementation of SSIM")

        try:
            from sewar.full_ref import ssim as ssim_sewar
            from sewar.utils import Filter

            def sewar_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # sewar operates on NumPy arrays and exposes filter settings explicitly.
                # By default, it uses a uniform filter, so this must be overridden.
                # Using ``mode="valid"`` keeps it aligned with the original MATLAB
                # implementation.
                score, _ = ssim_sewar(
                    reference.cpu().numpy(),
                    image.cpu().numpy(),
                    ws=self.kernel_size,
                    K1=self.k1,
                    K2=self.k2,
                    MAX=self.dynamic_range,
                    fltr_specs={"fltr": Filter.GAUSSIAN, "sigma": self.kernel_sigma, "ws": self.kernel_size},
                    mode="valid",
                )
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["sewar"] = sewar_ssim
        except Exception:
            logger.warning("sewar or its SSIM implementation is not available, skipping sewar implementation of SSIM")

        try:
            from monai.metrics import SSIMMetric

            monai_metric = SSIMMetric(
                spatial_dims=2,
                data_range=self.dynamic_range,
                kernel_type="gaussian",
                win_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
                k1=self.k1,
                k2=self.k2,
            )

            def monai_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # MONAI exposes SSIM as a metric object and expects NCHW tensors.
                # It uses valid padding, source of differences still unclear.
                return monai_metric(reference.unsqueeze(0).unsqueeze(0), image.unsqueeze(0).unsqueeze(0))

            implementations["monai"] = monai_ssim
        except Exception:
            logger.warning("monai or its SSIM implementation is not available, skipping monai implementation of SSIM")

        try:
            from deepinv.loss.metric import SSIM as DeepInvSSIM

            deepinv_metric = DeepInvSSIM(
                multiscale=False,
                max_pixel=self.dynamic_range,
                min_pixel=0.0,
                torchmetric_kwargs={
                    "gaussian_kernel": True,
                    "sigma": self.kernel_sigma,
                    "kernel_size": self.kernel_size,
                    "k1": self.k1,
                    "k2": self.k2,
                },
            )

            def deepinv_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # deepinv wraps torchmetrics-style SSIM for inverse-problems workflows,
                # so its results are expected to match torchmetrics very closely.
                return deepinv_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0))

            implementations["deepinv"] = deepinv_ssim
        except Exception:
            logger.warning(
                "deepinv or its SSIM implementation is not available, skipping deepinv implementation of SSIM"
            )

        try:
            from mondAI.metrics.third_party.medimetrics.ssim import SSIM as MediMetricsSSIM

            medimetric = MediMetricsSSIM()

            def medimetrics_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                score = medimetric.compute(
                    reference.cpu().numpy(),
                    image.cpu().numpy(),
                    data_range=self.dynamic_range,
                    kernel_size=self.kernel_size,
                    sigma=self.kernel_sigma,
                    k1=self.k1,
                    k2=self.k2,
                )
                return torch.tensor(score, device=image.device, dtype=image.dtype)

            implementations["medimetrics"] = medimetrics_ssim
        except Exception:
            logger.warning(
                "medimetrics or its SSIM implementation is not available, skipping medimetrics implementation of SSIM"
            )

        return implementations

    def __str__(self) -> str:
        """Full text representation of the metric."""
        arrow = self._arrow_indicating_optimum()
        return (
            f"{self.name} ({self.abbreviation}) {arrow} with parameters: "
            f"{self.k1=}, {self.k2=}, {self.kernel_size=}, {self.kernel_sigma=}, {self.dynamic_range=}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to SSIM."""
        if torch.any(image < 0) or torch.any(image > self.dynamic_range):
            raise ValueError(f"Input image contains pixel values outside the range [0, {self.dynamic_range}].")

        if torch.any(reference < 0) or torch.any(reference > self.dynamic_range):
            raise ValueError(f"Reference image contains pixel values outside the range [0, {self.dynamic_range}].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            if self.dynamic_range != 1.0:
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
