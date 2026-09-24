# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.piq import get_piq_vsi
from mondAI.utils.checks import check_rgb, check_value_range, warn_if_all_pixels_in_0_to_1_range
from mondAI.utils.signal_processing import gradient_map, subsample
from mondAI.utils.similarity_map import similarity_map

logger = get_logger()


class VSI(FullReferenceMetric):
    """Visual saliency-based index (VSI)

    This metric computes a visual saliency-induced index for perceptual image quality assessment.
    VSI is designed to capture the perceptual quality of images by incorporating visual saliency information,
    gradient magnitude similarity, and chromaticity similarity in a way that correlates well with human perception
    of image quality. In particular, VSI computes a visual saliency map for both the reference and distorted images
    and uses this for similarity computation but also uses these maps to weight the
    similarity comparisons between the images, giving more importance to regions that are more visually salient.
    It expects RGB images with pixel values in the range [0, 255], therefore input images are scaled
    from [0,1] to [0,255] by multiplying with a scaling factor of 255, and yields scores between 0 and 1
    where higher values indicate better perceptual quality. The metric is symmetric.

    Implementation based on original MATLAB implementation by Lin Zhang and
    piq's implementation, which is based on the original MATLAB code provided by the authors of VSI:
    https://github.com/photosynthesis-team/piq/blob/master/piq/vsi.py
    commit: 09aad9e1bde484dfbdc4b1bc20020711145636c5, License: Apache License 2.0
    Scores slightly deviate which is due to different interpolation behaviour in MATLAB's imresize function.

    Original publication:
    Zhang, Lin, Ying Shen, and Hongyu Li.
    "VSI: A visual saliency-induced index for perceptual image quality assessment."
    IEEE Transactions on Image processing 23.10 (2014): 4270-4281.

    """

    @property
    def name(self) -> str:
        return "Visual Saliency-based Index"

    @property
    def abbreviation(self) -> str:
        return "VSI"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def scaling_factor(self) -> float:
        return 255.0

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self,
        c1: float = 1.27,
        c2: float = 386.0,
        c3: float = 130.0,
        alpha: float = 0.40,
        beta: float = 0.020,
        sigma_d: float = 145.0,
        sigma_c: float = 0.001,
        omega_0: float = 0.0210,
        sigma_f: float = 1.34,
    ) -> None:
        """
        :param c1: Constant for numerical stability of visual saliency map, must be non-negative, default is 1.27.
        :type c1: float
        :param c2: Constant for gradient magnitude comparison, must be non-negative, default is 386.0.
        :type c2: float
        :param c3: Constant for chromaticity comparison, must be non-negative, default is 130.0.
        :type c3: float
        :param alpha: Weighting factor for the visual saliency map, must be non-negative, default is 0.40.
        :type alpha: float
        :param beta: Weighting factor for the gradient magnitude comparison, must be non-negative, default is 0.020.
        :type beta: float
        :param sigma_d: Coefficient for location weighting in visual saliency map, must be non-negative,
        default is 145.0.
        :type sigma_d: float
        :param sigma_c: Coefficient for color weighting in visual saliency map, must be non-negative, default is 0.001.
        :type sigma_c: float
        :param omega_0: The center frequency of the log-Gabor filter, must be non-negative, default is
            0.0210.
        :type omega_0: float
        :param sigma_f: The bandwidth of the log-Gabor filter, must be non-negative, default is 1.34.
        :type sigma_f: float
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.alpha = alpha
        self.beta = beta
        self.sigma_d = sigma_d
        self.sigma_c = sigma_c
        self.omega_0 = omega_0
        self.sigma_f = sigma_f

        if self.c1 < 0:
            raise ValueError(f"c1 must be non-negative, but got {self.c1}.")
        if self.c2 < 0:
            raise ValueError(f"c2 must be non-negative, but got {self.c2}.")
        if self.c3 < 0:
            raise ValueError(f"c3 must be non-negative, but got {self.c3}.")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, but got {self.alpha}.")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, but got {self.beta}.")
        if self.sigma_d < 0:
            raise ValueError(f"sigma_d must be non-negative, but got {self.sigma_d}.")
        if self.sigma_c < 0:
            raise ValueError(f"sigma_c must be non-negative, but got {self.sigma_c}.")
        if self.omega_0 < 0:
            raise ValueError(f"omega_0 must be non-negative, but got {self.omega_0}.")
        if self.sigma_f < 0:
            raise ValueError(f"sigma_f must be non-negative, but got {self.sigma_f}.")

        if (
            self.c1 != 1.27
            or self.c2 != 386.0
            or self.c3 != 130.0
            or self.alpha != 0.40
            or self.beta != 0.020
            or self.sigma_d != 145.0
            or self.sigma_c != 0.001
            or self.omega_0 != 0.0210
            or self.sigma_f != 1.34
        ):
            logger.warning(
                f"Using non-default values for c1={self.c1}, c2={self.c2}, c3={self.c3}, alpha={self.alpha}, "
                f"beta={self.beta}, sigma_d={self.sigma_d}, sigma_c={self.sigma_c}, omega_0={self.omega_0}, or "
                f"sigma_f={self.sigma_f} may lead "
                "to results that are not directly comparable to the original formulation of VSI, which uses "
                "c1=1.27, c2=386.0, c3=130.0, alpha=0.40, beta=0.020, sigma_d=145.0, sigma_c=0.001, omega_0=0.0210, "
                "and sigma_f=1.34. Please ensure that you understand the implications of changing these parameters on "
                "the metric's behavior and interpretability."
            )

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self._input_checks(image, reference)

        # If the images are identical, return a perfect score of 1.0
        if torch.equal(image, reference):
            return torch.ones((), device=image.device, dtype=image.dtype)

        saliency_map_image = self._saliency_map(
            image, sigma_d=self.sigma_d, sigma_c=self.sigma_c, omega_0=self.omega_0, sigma_f=self.sigma_f
        )
        saliency_map_reference = self._saliency_map(
            reference, sigma_d=self.sigma_d, sigma_c=self.sigma_c, omega_0=self.omega_0, sigma_f=self.sigma_f
        )

        # Convert to luminance and opponent color space
        rgb_to_lmn_matrix = torch.tensor(
            [[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]], device=image.device, dtype=image.dtype
        )
        image_lmn = torch.einsum("xc,chw->xhw", rgb_to_lmn_matrix, image)
        reference_lmn = torch.einsum("xc,chw->xhw", rgb_to_lmn_matrix, reference)

        # Downsample the images
        min_dimension = min(
            image.shape[self.expected_dimensions.index(dim)] for dim in (Dimension.HEIGHT, Dimension.WIDTH)
        )
        kernel_size = max(1, round(min_dimension / 256))
        if kernel_size // 2:
            padding = [kernel_size // 2, (kernel_size - 1) // 2, kernel_size // 2, (kernel_size - 1) // 2]
            image_lmn = torch.nn.functional.pad(image_lmn.unsqueeze(0), pad=padding, mode="replicate").squeeze(0)
            reference_lmn = torch.nn.functional.pad(reference_lmn.unsqueeze(0), pad=padding, mode="replicate").squeeze(
                0
            )
            saliency_map_image = torch.nn.functional.pad(
                saliency_map_image.unsqueeze(0), pad=padding, mode="replicate"
            ).squeeze(0)
            saliency_map_reference = torch.nn.functional.pad(
                saliency_map_reference.unsqueeze(0), pad=padding, mode="replicate"
            ).squeeze(0)
        image_lmn = subsample(image_lmn, kernel_size=kernel_size, channels=3)
        reference_lmn = subsample(reference_lmn, kernel_size=kernel_size, channels=3)
        saliency_map_image = subsample(saliency_map_image, kernel_size=kernel_size, channels=1)
        saliency_map_reference = subsample(saliency_map_reference, kernel_size=kernel_size, channels=1)

        # Calculate gradient maps
        sharr_kernel = (
            torch.outer(
                torch.tensor([3.0, 10.0, 3.0]) / 16,
                torch.tensor([1.0, 0.0, -1.0]),
            )
            .double()
            .to(image.device)
        )

        gradient_map_image = gradient_map(image_lmn[0], sharr_kernel)
        gradient_map_reference = gradient_map(reference_lmn[0], sharr_kernel)

        # Calculate VSI
        visual_saliency_similarity = similarity_map(saliency_map_image, saliency_map_reference, self.c1)
        gradient_similarity = similarity_map(gradient_map_image, gradient_map_reference, self.c2)
        weight = torch.maximum(saliency_map_image, saliency_map_reference)

        similarity_m_channel = similarity_map(image_lmn[1], reference_lmn[1], self.c3).to(torch.complex128)
        similarity_n_channel = similarity_map(image_lmn[2], reference_lmn[2], self.c3).to(torch.complex128)

        similarity = (
            gradient_similarity**self.alpha
            * visual_saliency_similarity
            * ((similarity_m_channel * similarity_n_channel) ** self.beta).real
            * weight
        )

        return similarity.sum() / weight.sum()

    def _saliency_map(
        self,
        image: torch.Tensor,
        sigma_d: float = 145.0,
        sigma_c: float = 0.001,
        omega_0: float = 0.0210,
        sigma_f: float = 1.34,
    ) -> torch.Tensor:
        """
        Saliency Detection by combining Simple Priors (frequency, location, color).
        Default parameters as defined in original paper:

        Zhang, Lin, Zhongyi Gu, and Hongyu Li.
        "SDSP: a novel saliency detection method by combining simple priors."
        2013 IEEE international conference on image processing. IEEE, 2013.

        :param image: Input RGB image tensor with shape (3, H, W) and pixel values in the range [0, 255].
        :type image: torch.Tensor
        :param sigma_d: Coefficient for location weighting, default is 145.0.
        :type sigma_d: float
        :param sigma_c: Coefficient for color weighting, default is 0.001.
        :type sigma_c: float
        :param omega_0: The center frequency of the log-Gabor filter, must be non-negative, default is 0.0210.
        :type omega_0: float
        :param sigma_f: The bandwidth of the log-Gabor filter, must be non-negative, default is 1.34.
        :type sigma_f: float
        :return: Saliency map tensor with shape (1, H, W).
        :rtype: torch.Tensor
        """

        target_size = (256, 256)
        image_resized = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        # convert to lab color space
        image_resized = self._rgb_to_lab(image_resized)

        IMAGE = torch.fft.fft2(image_resized)
        x = torch.arange(-target_size[0] / 2, target_size[0] / 2, device=image.device, dtype=image.dtype)
        y = torch.arange(-target_size[1] / 2, target_size[1] / 2, device=image.device, dtype=image.dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        log_gabor_filter = self._log_gabor_filters(
            xx / target_size[0], yy / target_size[1], omega_0=omega_0, sigma_f=sigma_f
        )
        filtered = torch.fft.ifft2(IMAGE * log_gabor_filter).real
        frequency_prior = torch.sqrt(torch.sum(filtered**2, dim=0))

        # central areas will have bias towards attention
        coordinates = torch.stack((xx, yy), dim=0)
        coordinates = coordinates + 1
        location_prior = torch.exp(-torch.sum((coordinates**2), dim=0) / sigma_d**2)

        # warm colors have a bias towards attention
        normalized_A_channel = (image_resized[1] - image_resized[1].min()) / (
            image_resized[1].max() - image_resized[1].min()
        )
        normalized_B_channel = (image_resized[2] - image_resized[2].min()) / (
            image_resized[2].max() - image_resized[2].min()
        )
        lab_distance_squared = normalized_A_channel**2 + normalized_B_channel**2
        color_prior = 1 - torch.exp(-lab_distance_squared / (sigma_c**2))

        saliency_map = frequency_prior * location_prior * color_prior
        saliency_map = torch.nn.functional.interpolate(
            saliency_map.unsqueeze(0).unsqueeze(0), size=image.shape[1:3], mode="bilinear", align_corners=True
        ).squeeze(0)
        return (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min() + torch.finfo(image.dtype).eps
        )

    def _rgb_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB image to CIE Lab color space.

        :param image: Input RGB image tensor with shape (3, H, W) and pixel values in
            the range [0, 255].
        :type image: torch.Tensor
        :return: Image tensor in Lab color space with shape (3, H, W).
        :rtype: torch.Tensor

        """
        normalized_image = image / 255.0

        normalized_image = torch.where(
            normalized_image <= 0.04045, normalized_image / 12.92, torch.pow((normalized_image + 0.055) / 1.055, 2.4)
        )

        rgb_to_xyz_matrix = torch.tensor(
            [[0.4124564, 0.2126729, 0.0193339], [0.3575761, 0.7151522, 0.119192], [0.1804375, 0.0721750, 0.9503041]],
            device=image.device,
            dtype=image.dtype,
        )
        xyz_image = torch.einsum("xc,chw->xhw", rgb_to_xyz_matrix.T, normalized_image)

        # D50 white point
        white_point = torch.tensor([0.9642119944211994, 1, 0.8251882845188288], device=image.device, dtype=image.dtype)
        xyz_image_normalized = xyz_image / white_point[:, None, None]

        f_xyz = torch.where(
            xyz_image_normalized > 0.008856,
            torch.pow(xyz_image_normalized, 1.0 / 3.0),
            (903.3 * xyz_image_normalized + 16.0) / 116.0,
        )

        return torch.stack(
            (
                116.0 * f_xyz[1] - 16.0,
                500.0 * (f_xyz[0] - f_xyz[1]),
                200.0 * (f_xyz[1] - f_xyz[2]),
            ),
            dim=0,
        )

    def _log_gabor_filters(
        self, xx: torch.Tensor, yy: torch.Tensor, omega_0: float = 0.0210, sigma_f: float = 1.34
    ) -> torch.Tensor:
        """Create log-Gabor filters in the frequency domain.

        :param xx: 2D tensor representing the x-coordinates in the frequency domain,
            normalized to [-0.5, 0.5].
        :type xx: torch.Tensor
        :param yy: 2D tensor representing the y-coordinates in the frequency domain,
            normalized to [-0.5, 0.5].
        :type yy: torch.Tensor
        :param omega_0: The center frequency of the log-Gabor filter, default is
            0.0210.
        :type omega_0: float
        :param sigma_f: The bandwidth of the log-Gabor filter, default is 1.34.
        :type sigma_f: float
        :return: A 2D tensor representing the log-Gabor filter in the frequency domain.
        :rtype: torch.Tensor

        """
        radius = torch.sqrt(xx**2 + yy**2)
        mask = radius <= 0.5
        radius = radius * mask
        radius = torch.fft.ifftshift(radius)
        radius[0, 0] = 1.0
        log_gabor = torch.exp(-(torch.log(radius / omega_0) ** 2) / (2 * sigma_f**2))
        log_gabor[0, 0] = 0.0
        return log_gabor

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        """Override this method in subclasses to register other implementations of the
        metric for comparison. Use the `_register_implementation` helper method to add
        implementations to the internal dictionary. These implementations will be used
        when compare_implementations is True.

        Store reference implementations in the `third_party` submodule of the metrics module,
        and import them here to register them for comparison.

        """
        self._register_implementation(
            implementations,
            "piq",
            get_piq_vsi(
                c1=self.c1,
                c2=self.c2,
                c3=self.c3,
                alpha=self.alpha,
                beta=self.beta,
                omega_0=self.omega_0,
                sigma_d=self.sigma_d,
                sigma_c=self.sigma_c,
                sigma_f=self.sigma_f,
            ),
        )
        # self._register_implementation(implementations, "piqa", get_piqa_vsi())
        # PIQA's VSI implementation is not working as there is a shape mismatch in the saliency map computation

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} "
            f"with c1={self.c1}, c2={self.c2}, c3={self.c3}, alpha={self.alpha}, beta={self.beta}, "
            f"sigma_d={self.sigma_d}, sigma_c={self.sigma_c}, omega_0={self.omega_0}, sigma_f={self.sigma_f}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to VSI, such as checking for valid pixel value
        ranges and dimensions.

        Warns if all pixel values in both image and reference are in the range [0, 1],
        which may indicate that the images are not correctly scaled for VSI, which
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
            VSI.

        """
        # Input checks specific to VSI
        check_value_range(image, 0, 255)
        check_value_range(reference, 0, 255, reference=True)

        warn_if_all_pixels_in_0_to_1_range(image, reference)

        check_rgb(self.expected_dimensions, image, self.abbreviation)
