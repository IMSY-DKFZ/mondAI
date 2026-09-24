# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable
from functools import partial

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.others import get_pytorch_iwssim
from mondAI.metrics.third_party.piq import get_piq_iwssim
from mondAI.utils.checks import check_value_range
from mondAI.utils.signal_processing import convolve2d
from mondAI.utils.similarity_map import ssim_and_cs_maps

logger = get_logger()


class IWSSIM(FullReferenceMetric):
    r"""Information Content Weighted (Multi-Scale) Structural Similarity Index (IW-
    SSIM).

    The Information Content Weighted (Multi-Scale) Structural Similarity Index (IW-SSIM)
    extends multi-scale SSIM by weighting local similarity values according to the
    estimated local information content in the images. Intuitively, distortions
    in perceptually informative regions contribute more strongly to the final score than
    distortions in flat or uninformative regions.

    This implementation computes SSIM-like maps at multiple scales, estimates local
    information weights from image pyramids, and then aggregates the
    weighted multiscale scores using the standard MS-SSIM-style product formula.

    A compact form of the final score is

    .. math::
        \operatorname{IW\text{-}SSIM}(x, y) = \prod_{j=1}^{M} \operatorname{IWCS}_j(x, y)^{\beta_j},

    where :math:`M` is the number of scales, :math:`\beta_j` are scale weights, and
    :math:`\operatorname{IWCS}_j` is the information-weighted mean of the local
    similarity map at scale :math:`j`.

    Note that this metric is not symmetric, as the information content weighting is based on the reference image.

    Implementation adapted from the original MATLAB implementation by Zhou Wang
    (https://ece.uwaterloo.ca/~z70wang/research/iwssim/) and
    PIQ (https://github.com/photosynthesis-team/piq/blob/master/piq/iw_ssim.py commit:
    213a46687ad99098f274784e61d92c5144a94a68, license: Apache 2.0).

    Original publication:
    Z. Wang, and L. Qiang,
    "Information content weighting for perceptual image quality assessment",
    IEEE Transactions on Image Processing, vol. 20, no. 5, pp. 1185-1198, 2011,
    doi: 10.1109/TIP.2010.2092435

    Differences to other implementations:

    - this implementation is grayscale-only,
    - it exposes the main IW-SSIM parameters used across common implementations,
    - it expects images in ``[0, dynamic_range]`` and defaults to ``dynamic_range=255``.

    """

    DEFAULT_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    @property
    def name(self) -> str:
        return "Information Content Weighted Structural Similarity Index"

    @property
    def abbreviation(self) -> str:
        return "IW-SSIM"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def scaling_factor(self) -> float:
        return 255.0

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
        information_content_weighting: bool = True,
        block_size: int = 3,
        include_parent: bool = True,
        sigma_n_squared: float = 0.4,
    ) -> None:
        """Initialize IW-SSIM with common default parameter values.

        :param k1: First stability constant coefficient, default is 0.01, must be non-negative.
        :type k1: float
        :param k2: Second stability constant coefficient, default is 0.03, must be non-negative.
        :type k2: float
        :param kernel_size: Size of the Gaussian window, default is 11, must be odd and at least 3
        :type kernel_size: int
        :param kernel_sigma: Standard deviation of the Gaussian window, default is 1.5, must be positive.
        :type kernel_sigma: float
        :param dynamic_range: Dynamic range ``L`` of the images, default is 255.0, must be positive.
        :type dynamic_range: float
        :param scales: Number of scales, needs to be a positive integer,
         setting this to 1 results in scores equal to SSIM, default is 5.
        :type scales: int
        :param weights: Weights for each scale, default to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
          which were derived emperically by the original authors.
          The length of weights must match the number of scales, and they will be normalized.
        :type weights: tuple[float, ...]
        :param information_content_weighting: Whether to apply information weighting or not, default is True.
        :type information_content_weighting: bool
        :param block_size: Block size of spatial neighborhood for local information content estimation, needs to be odd,
          default is 3.
        :type block_size: int
        :param include_parent: Whether to include the parent neighbor, default is True.
        :type include_parent: bool
        :param sigma_n_squared: Noise variance parameter for information content weighting, default is 0.4,
        must be non-negative.
        :type sigma_n_squared: float
        :raises ValueError: If any of the parameters are out of their valid ranges or conditions,
        such as negative values for k1, k2, or sigma_n_squared, non-positive values for kernel_sigma or dynamic_range,
          even values for kernel_size or block_size, scales less than 1, weights that do not match the number of scales
            or sum to zero.

        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.dynamic_range = dynamic_range
        self.scales = scales
        self.weights = weights
        self.information_content_weighting = information_content_weighting
        self.block_size = block_size
        self.include_parent = include_parent
        self.sigma_n_squared = sigma_n_squared

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
        if len(self.weights) < self.scales:
            raise ValueError(
                f"weights must contain at least as many entries as scales, but got {len(self.weights)} < {self.scales}."
            )
        if sum(self.weights[: self.scales]) == 0:
            raise ValueError(f"weights must not sum to zero, but got {sum(self.weights[: self.scales])}.")
        if self.block_size < 1:
            raise ValueError(f"block_size must be at least 1, but got {self.block_size}.")
        if self.block_size % 2 == 0:
            raise ValueError(f"block_size must be odd, but got {self.block_size}.")
        if self.sigma_n_squared < 0:
            raise ValueError(f"sigma_n_squared must be non-negative, but got {self.sigma_n_squared}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the metric between image and reference. Inputs must be at least
         `kernel_size` * (2 ^ (`scales` - 1)) pixels in height and width to allow for
         the required number of downsampling steps and kernel size.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :return: The computed metric score.
        :rtype: torch.Tensor

        """
        self._input_checks(image, reference)

        # If the images are identical, return a perfect score of 1.0
        if torch.equal(image, reference):
            return torch.ones((), device=image.device, dtype=image.dtype)

        weights = torch.tensor(self.weights[: self.scales], device=image.device, dtype=image.dtype)
        weights = weights / weights.sum()

        iw_pad = math.ceil((self.kernel_size - 1) / 2) - math.floor((self.block_size - 1) / 2)

        weighted_mean_contrast_and_structure = image.new_zeros(self.scales)

        # Laplacian pyramids extracted from images, adapted from pyrtools package
        image_pyramid = self._get_pyramid(image)
        reference_pyramid = self._get_pyramid(reference)

        for scale in range(self.scales):
            ssim_map, cs_map = ssim_and_cs_maps(
                image_pyramid[scale],
                reference_pyramid[scale],
                kernel_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
                dynamic_range=self.dynamic_range,
                k1=self.k1,
                k2=self.k2,
            )

            if self.information_content_weighting and scale < self.scales - 1:
                # estimate information content weight map from reference and image pyramids
                iw_map = self._information_weight_map(
                    reference_pyramid[scale],
                    image_pyramid[scale],
                    reference_pyramid[scale + 1] if scale < self.scales - 2 else None,
                )
                if iw_pad > 0:
                    iw_map = iw_map[iw_pad:-iw_pad, iw_pad:-iw_pad]

                # compute weighted mean of contrast-structure map
                weighted_score = (cs_map * iw_map).sum() / (iw_map.sum() + torch.finfo(image.dtype).eps)
            else:
                # no information content weighting at the coarsest scale, or if disabled
                weighted_score = ssim_map.mean() if scale == self.scales - 1 else cs_map.mean()

            weighted_mean_contrast_and_structure[scale] = weighted_score

        return torch.prod(torch.abs(weighted_mean_contrast_and_structure) ** weights)

    def _information_weight_map(
        self, reference: torch.Tensor, image: torch.Tensor, parent: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the local information content weight for the given reference and
        image pyramids.

        :param reference: Reference / undistorted image
        :type reference: torch.Tensor
        :param image: Distorted image
        :type image: torch.Tensor
        :param parent: Parent pyramid level of the reference image, which can be used
            for additional information content estimation if include_parent is True. If
            not provided, the information content estimation will be based solely on
            the current level. Default is None.
        :type parent: torch.Tensor | None, optional
        :return: Information content weight map
        :rtype: torch.Tensor

        """

        tolerance = 1e-15  # as in original MATLAB implementation

        kernel = image.new_ones((self.block_size, self.block_size)) / float(self.block_size**2)

        mean_image = convolve2d(image, kernel, padding="same")
        mean_reference = convolve2d(reference, kernel, padding="same")
        covariance = convolve2d(image * reference, kernel, padding="same") - mean_image * mean_reference
        variance_image = convolve2d(image**2, kernel, padding="same") - mean_image**2
        variance_reference = convolve2d(reference**2, kernel, padding="same") - mean_reference**2

        variance_image = torch.clamp(variance_image, min=0.0)
        variance_reference = torch.clamp(variance_reference, min=0.0)

        gain = covariance / (variance_reference + tolerance)
        sigma_v_squared = variance_image - gain * covariance
        gain[variance_reference < tolerance] = 0.0
        sigma_v_squared[variance_reference < tolerance] = variance_image[variance_reference < tolerance]
        variance_reference[variance_reference < tolerance] = 0.0
        gain[variance_image < tolerance] = 0.0
        sigma_v_squared[variance_image < tolerance] = 0.0

        # Parent pyramid layer
        if self.include_parent and parent is not None:

            def _image_enlarge2d(img: torch.Tensor) -> torch.Tensor:
                """Custom 2D image enlargement function that mimics the behavior of the
                original MATLAB code, which does upsampling by a factor of 4x-3 with
                bilinear interpolation, handles boundaries with difference padding, and
                then downsamples by a factor of 2."""
                upsampled_inner = torch.nn.functional.interpolate(
                    img.unsqueeze(0).unsqueeze(0), size=(4 * img.shape[0] - 3, 4 * img.shape[1] - 3), mode="bilinear"
                )
                upsampled = img.new_zeros((4 * img.shape[0] - 1, 4 * img.shape[1] - 1))
                upsampled[1:-1, 1:-1] = upsampled_inner

                # Handle boundaries with difference padding as in the original MATLAB code
                upsampled[0, :] = 2 * upsampled[1, :] - upsampled[2, :]
                upsampled[-1, :] = 2 * upsampled[-2, :] - upsampled[-3, :]
                upsampled[:, 0] = 2 * upsampled[:, 1] - upsampled[:, 2]
                upsampled[:, -1] = 2 * upsampled[:, -2] - upsampled[:, -3]
                return upsampled[::2, ::2]

            parent = _image_enlarge2d(parent)[: reference.shape[-2], : reference.shape[-1]]

        # group neighboring pixels into blocks and compute local information content
        neighborhood_size = self.block_size**2 + (self.include_parent and parent is not None)
        block_size_half = int((self.block_size - 1) / 2)
        neighborhood_height = (reference.shape[-2] - self.block_size) + 1  # discard outer coefficients
        neighborhood_width = (reference.shape[-1] - self.block_size) + 1
        neighborhoods = image.new_zeros((neighborhood_height * neighborhood_width, neighborhood_size))

        neighbor = 0
        for ny in range(-block_size_half, block_size_half + 1):
            for nx in range(-block_size_half, block_size_half + 1):
                shifted_reference = torch.roll(reference, shifts=(ny, nx), dims=(-2, -1))
                neighborhoods[:, neighbor] = shifted_reference[
                    block_size_half : block_size_half + neighborhood_height,
                    block_size_half : block_size_half + neighborhood_width,
                ].flatten()
                neighbor += 1
        if self.include_parent and parent is not None:
            neighborhoods[:, -1] = parent[
                block_size_half : block_size_half + neighborhood_height,
                block_size_half : block_size_half + neighborhood_width,
            ].flatten()

        # positive-definite covariance matrix
        covariance_U = torch.matmul(neighborhoods.T, neighborhoods) / (neighborhood_height * neighborhood_width)
        eigen_values, eigen_vectors = torch.linalg.eigh(covariance_U, UPLO="U")

        # correct negative eigenvaluees
        non_zero_eigen_values = eigen_values * (eigen_values > 0)
        corrected_eigen_values = (
            torch.diag(non_zero_eigen_values)
            * torch.sum(eigen_values)
            / (torch.sum(non_zero_eigen_values) + torch.sum(non_zero_eigen_values == 0))
        )

        covariance_U = torch.matmul(torch.matmul(eigen_vectors, corrected_eigen_values), eigen_vectors.T)
        covariance_U_inv = torch.inverse(covariance_U)
        s_squared = torch.matmul(neighborhoods, covariance_U_inv) * neighborhoods / neighborhood_size
        s_squared = torch.sum(s_squared, dim=1).reshape(neighborhood_height, neighborhood_width)
        gain = gain[
            block_size_half : block_size_half + neighborhood_height,
            block_size_half : block_size_half + neighborhood_width,
        ]
        sigma_v_squared = sigma_v_squared[
            block_size_half : block_size_half + neighborhood_height,
            block_size_half : block_size_half + neighborhood_width,
        ]

        # Calculate mutual information
        weight = torch.sum(
            torch.log2(
                1
                + (
                    (sigma_v_squared[:, :, None] + (1 + gain[:, :, None] ** 2) * self.sigma_n_squared)
                    * s_squared[:, :, None]
                    * eigen_values
                    + self.sigma_n_squared * sigma_v_squared[:, :, None]
                )
                / (self.sigma_n_squared**2)
            ),
            dim=-1,
        )
        weight[weight < tolerance] = 0.0
        return weight

    def _get_pyramid(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        """Get Laplacian image pyramid for the given image with a binomial filter and
        reflection padding.

        Implementation is based on pyrtools package: https://github.com/LabForComputationalVision/pyrtools, license: MIT

        """

        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=image.dtype, device=image.device)
        kernel = kernel / 16.0 * 2**0.5

        def build_next_level(image: torch.Tensor) -> torch.Tensor:
            """Build the next level of the pyramid."""

            def corrDn(
                image: torch.Tensor, filt: torch.Tensor, padding: tuple[int, int, int, int], step: tuple[int, int]
            ) -> torch.Tensor:
                """Correlated image with filter and downsample."""

                # Apply convolution
                image = torch.nn.functional.pad(image[None, None, :, :], padding, mode="reflect")
                convolved = torch.nn.functional.conv2d(image, filt)

                # Downsample
                return convolved[:, :, :: step[0], :: step[1]].squeeze()

            if image.shape[0] == 1:
                result = corrDn(image=image, filt=kernel[None, None, None, :], padding=(2, 2, 0, 0), step=(1, 2))
            elif image.shape[1] == 1:
                result = corrDn(image=image, filt=kernel[None, None, :, None], padding=(0, 0, 2, 2), step=(2, 1))
            else:
                result = corrDn(image=image, filt=kernel[None, None, None, :], padding=(2, 2, 0, 0), step=(1, 2))
                result = corrDn(image=result, filt=kernel[None, None, :, None], padding=(0, 0, 2, 2), step=(2, 1))
            return result

        def reconstruct_previous_level(image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
            """Reconstruct the previous level of the pyramid from a given image to the
            output size."""

            def upConv(
                image: torch.Tensor,
                filt: torch.Tensor,
                padding: tuple[int, int, int, int],
                step: tuple[int, int],
                stop: tuple[int, int],
            ) -> torch.Tensor:
                """Upsample via zero-insertion + convolution."""

                # Create an upsampled tensor with zero-insertion
                upsampled = image.new_zeros((image.shape[0] * step[0], image.shape[1] * step[1]))
                upsampled[:: step[0], :: step[1]] = image

                # Apply convolution
                upsampled = torch.nn.functional.pad(upsampled[None, None, :, :], padding, mode="reflect")
                upsampled = torch.nn.functional.conv2d(upsampled, filt)

                # Crop to the desired output size
                return upsampled[:, :, : stop[0], : stop[1]].squeeze()

            if image.shape[0] == 1:
                result = upConv(
                    image=image, filt=kernel[None, None, None, :], padding=(2, 2, 0, 0), step=(1, 2), stop=output_size
                )
            elif image.shape[1] == 1:
                result = upConv(
                    image=image, filt=kernel[None, None, :, None], padding=(0, 0, 2, 2), step=(2, 1), stop=output_size
                )
            else:
                result = upConv(
                    image=image,
                    filt=kernel[None, None, None, :],
                    padding=(2, 2, 0, 0),
                    step=(1, 2),
                    stop=(image.shape[0], output_size[1]),
                )
                result = upConv(
                    image=result, filt=kernel[None, None, :, None], padding=(0, 0, 2, 2), step=(2, 1), stop=output_size
                )
            return result

        pyramid = {}
        for level in range(self.scales - 1):
            image_next = build_next_level(image)
            image_reconstructed = reconstruct_previous_level(image_next, output_size=image.shape)
            image_residual = image - image_reconstructed
            pyramid[level] = image_residual
            image = image_next
        pyramid[self.scales - 1] = image

        return pyramid

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(
            implementations,
            "piq",
            partial(
                get_piq_iwssim,
                self.dynamic_range,
                self.kernel_size,
                self.kernel_sigma,
                self.weights,
                self.k1,
                self.k2,
                self.include_parent,
                self.block_size,
                self.sigma_n_squared,
            ),
        )
        self._register_implementation(
            implementations,
            "pytorch",
            partial(
                get_pytorch_iwssim,
                self.information_content_weighting,
                self.scales,
                self.block_size,
                self.block_size,
                self.include_parent,
                self.sigma_n_squared,
            ),
        )

    def __str__(self) -> str:
        """Full text representation of the metric."""
        arrow = self._arrow_indicating_optimum()
        return (
            f"{self.name} ({self.abbreviation}) {arrow} with parameters: "
            f"{self.k1=}, {self.k2=}, {self.kernel_size=}, {self.kernel_sigma=}, {self.dynamic_range=}, "
            f"{self.scales=}, {self.weights=}, {self.information_content_weighting=}, {self.block_size=}, "
            f"{self.include_parent=}, {self.sigma_n_squared=}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to IW-SSIM."""
        check_value_range(image, 0, self.dynamic_range)
        check_value_range(reference, 0, self.dynamic_range, reference=True)

        if image.shape[-2] < self.kernel_size or image.shape[-1] < self.kernel_size:
            raise ValueError(
                f"Images have spatial dimensions {image.shape[-2:]} which are smaller than the required window size "
                f"{self.kernel_size}x{self.kernel_size} for IW-SSIM."
            )

        minimum_image_width = self.kernel_size * (2 ** (self.scales - 1))
        if min(image.shape[-2], image.shape[-1]) < minimum_image_width:
            raise ValueError(
                f"Image size too small for {self.scales} scale IW-SSIM evaluation with kernel size {self.kernel_size}."
            )
