from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.utils.signal_processing import convolve2d, gaussian_filter_kernel

logger = get_logger()


class VIFP(FullReferenceMetric):
    """Visual Information Fidelity in the Pixel domain (VIFP) computes the ratio of the
    information that can be extracted from the distorted image to the information that
    can be extracted from the reference image, based on a natural scene statistical
    model and a human visual system (HVS) model. The VIFP metric is designed to measure
    the perceptual quality of images by quantifying how much visual information is
    preserved in the distorted image compared to the reference image.

    This implementation computes VIFP in the pixel domain. The VIF measure was originally defined in the wavelet
    domain, but the authors later released code to compute an approximation of VIF in the pixel domain, which is what
    this implementation is based on. The pixel domain version of VIFP is easier to compute than the wavelet domain
    version.

    VIFP expects grayscale images with a value range of [0, 255]. VIFP requires that input images have spatial
    dimensions of at least 41x41 pixels to ensure that they do not become too small after downsampling 4 times in the
    VIFP computation. It is not symmetric, meaning that the order of the input images matters. It yields scores mostly
    between 0 and 1, where higher values indicate better perceptual quality. However, it can yield values greater
    than 1 for predicted images with higher contrast than the original reference image. Other libraries such as piq
    also compute VIFP in the pixel domain, but they take the luminance channel of RGB images instead of computing it
    channel-wise, which is a different approach than this implementation.

    Implementation adapted from PIQ (https://github.com/photosynthesis-team/piq/blob/master/piq/vif.py commit:
    213a46687ad99098f274784e61d92c5144a94a68, License: Apache License 2.0) and MATLAB code provided by the
    original authors of VIFP (https://live.ece.utexas.edu/research/Quality/VIF.htm, License: BSD 3-Clause License).

    Original publication: H.R. Sheikh.and A.C. Bovik, "Image information and visual quality,"
    IEEE Transactions on Image Processing , vol.15, no.2,pp. 430- 444, Feb. 2006.

    """

    @property
    def name(self) -> str:
        return "Visual Information Fidelity in Pixel domain"

    @property
    def abbreviation(self) -> str:
        return "VIFP"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, sigma_n_squared: float = 2.0) -> None:
        """Initialize VIFP.

        :param sigma_n_squared: The variance of the visual noise in the image (HVS
            model parameter). Must be non-negative. Default is 2.0.
        :type sigma_n_squared: float
        :raises ValueError: If sigma_n_squared is negative.

        """
        super().__init__()
        self.sigma_n_squared = sigma_n_squared

        if self.sigma_n_squared < 0:
            raise ValueError("sigma_n_squared must be non-negative.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the metric between image and reference. Images must be at least
        41x41 pixels in size.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :return: The computed metric score.
        :rtype: torch.Tensor

        """

        self._input_checks(image, reference)

        EPSILON = 1e-10  # Small constant to prevent division by zero as defined in original MATLAB code

        numerator = 0.0
        denominator = 0.0

        # Downsample and compute VIFP at four scales
        for scale in range(4):
            kernel_size = 2 ** (4 - scale) + 1
            kernel = gaussian_filter_kernel(kernel_size, sigma=kernel_size / 5, device=image.device, dtype=image.dtype)

            # Downsampling
            if scale > 0:
                image = convolve2d(image, kernel, padding="valid")[::2, ::2]
                reference = convolve2d(reference, kernel, padding="valid")[::2, ::2]

            mu_reference = convolve2d(reference, kernel, padding="valid")
            mu_image = convolve2d(image, kernel, padding="valid")

            mu_reference_squared = mu_reference**2
            mu_image_squared = mu_image**2
            mu_both = mu_reference * mu_image

            sigma_reference_squared = convolve2d(reference**2, kernel, padding="valid") - mu_reference_squared
            sigma_image_squared = convolve2d(image**2, kernel, padding="valid") - mu_image_squared
            sigma_both = convolve2d(reference * image, kernel, padding="valid") - mu_both

            sigma_reference_squared = torch.clamp(sigma_reference_squared, min=0)
            sigma_image_squared = torch.clamp(sigma_image_squared, min=0)

            gain = sigma_both / (sigma_reference_squared + EPSILON)
            sigma_v_squared = sigma_image_squared - gain * sigma_both

            # Masking
            gain[sigma_reference_squared < EPSILON] = 0
            sigma_v_squared[sigma_reference_squared < EPSILON] = sigma_image_squared[sigma_reference_squared < EPSILON]
            sigma_reference_squared[sigma_reference_squared < EPSILON] = 0

            gain[sigma_image_squared < EPSILON] = 0
            sigma_v_squared[sigma_image_squared < EPSILON] = 0

            sigma_v_squared[gain < 0] = sigma_image_squared[gain < 0]
            gain = torch.clamp(gain, min=0.0)
            sigma_v_squared[sigma_v_squared <= EPSILON] = EPSILON

            numerator += torch.sum(
                torch.log10(1 + gain**2 * sigma_reference_squared / (sigma_v_squared + self.sigma_n_squared))
            )
            denominator += torch.sum(torch.log10(1 + sigma_reference_squared / self.sigma_n_squared))

        return numerator / denominator

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the metric. This will be
        used when compare_implementations is True to compute the metric using different
        libraries or implementations for comparison.

        :return: A dictionary where the keys are the names of the libraries or implementations, and the values are
        callables that compute the metric using those implementations.
        :rtype: dict[str, callable[..., torch.Tensor]]

        """

        implementations = {}

        # PIQ implementation
        try:
            from piq import vif_p as vifp_piq

            def piq_vifp(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # piq's implementation expects inputs with shape (N, C, H, W) and
                # data_range parameter should correspond to pixel value range

                return vifp_piq(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    data_range=255.0,
                    sigma_n_sq=self.sigma_n_squared,
                )

            implementations["piq"] = piq_vifp

        except ImportError:
            logger.warning("piq or its VIFP implementation is not available, skipping piq implementation of VIFP")

        # torchmetrics implementation
        try:
            from torchmetrics.functional.image.vif import visual_information_fidelity as vifp_torchmetrics

            def torchmetrics_vifp(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # torchmetrics' implementation expects inputs with shape (N, C, H, W)

                return vifp_torchmetrics(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    sigma_n_sq=self.sigma_n_squared,
                )

            implementations["torchmetrics"] = torchmetrics_vifp

        except ImportError:
            logger.warning(
                "torchmetrics or its VIFP implementation is not available, skipping torchmetrics implementation of VIFP"
            )

        # sewar implementation
        try:
            from sewar.full_ref import vifp as vifp_sewar

            def sewar_vifp(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # sewar's implementation expects numpy arrays

                vifp_numpy = vifp_sewar(image.cpu().numpy(), reference.cpu().numpy(), sigma_nsq=self.sigma_n_squared)
                return torch.tensor(vifp_numpy).to(image.device).to(image.dtype)

            implementations["sewar"] = sewar_vifp

        except ImportError:
            logger.warning("sewar or its VIFP implementation is not available, skipping sewar implementation of VIFP")

        return implementations

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        arrow = self._arrow_indicating_optimum()
        return f"{self.name} ({self.abbreviation}) {arrow} with sigma_n_squared={self.sigma_n_squared}"

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to VIFP, such as checking for valid pixel
        value ranges.

        Warns if all pixel values in both image and reference are in the range [0, 1],
        which may indicate that the images are not correctly scaled for VIFP, which
        expects pixel values in the range [0, 255].

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :raises ValueError: If the input image contains pixel values outside the range
            [0, 255].
        :raises ValueError: If the reference image contains pixel values outside the
            range [0, 255].
        :raises ValueError: If the spatial dimensions of the images are smaller than
            41x41 pixels.

        """
        # Input checks specific to VIFP
        if torch.any(image < 0) or torch.any(image > 255):
            raise ValueError("Input image contains pixel values outside the range [0, 255].")

        if torch.any(reference < 0) or torch.any(reference > 255):
            raise ValueError("Reference image contains pixel values outside the range [0, 255].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            logger.warning(
                "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "VIFP expects pixel values in the range [0, 255]. Please ensure that your input images are "
                "correctly scaled for accurate computation of VIFP."
            )

        MINIMUM_REQUIRED_SIZE = 41
        if image.shape[-2] < MINIMUM_REQUIRED_SIZE or image.shape[-1] < MINIMUM_REQUIRED_SIZE:
            raise ValueError(
                f"Images have spatial dimensions {image.shape[-2:]} which are smaller than the minimum required size of"
                f"{MINIMUM_REQUIRED_SIZE}x{MINIMUM_REQUIRED_SIZE} for VIFP. This size is required to ensure that the"
                "images do not become too small after downsampling 4 times in the VIFP computation."
            )
