from collections.abc import Callable
from pathlib import Path

import numpy
import torch
from scipy import stats
from scipy.io import loadmat

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.no_reference.base import NoReferenceMetric
from mondAI.metrics.third_party.medimetrics import get_medimetrics_niqe
from mondAI.utils.checks import check_value_range, warn_if_all_pixels_in_0_to_1_range
from mondAI.utils.signal_processing import convolve2d, gaussian_filter_kernel

logger = get_logger()


class NIQE(NoReferenceMetric):
    """Natural Image Quality Evaluator (NIQE) index which is based on the statistical
    features of natural images and measures the deviation of the input image from these
    natural image statistics. The NIQE index is designed to be "completely blind,"
    meaning it does not rely on any specific distortion model or reference image,
    making it applicable to a wide range of image quality assessment tasks.

    Asymmetric generalized Gaussian distribution features are computed on mean subtracted and contrast normalized
    patches and then fitted to a multivariate Gaussian distribution.
    The distance between the parameters of the fitted distribution and the parameters of a
    distribution fitted to natural images is then computed as the quality score. The distance is computed using the
    Mahalanobis distance, which takes into account the covariance of the features.

    This implemtentation and it's scores deviate from the original MATLAB implementation due to numerical differences
    in the image filtering and resize functions between MATLAB and pytorch which are enhanced by the
    computation of this metric. This metric expects grayscale images. Apply an `rgb2gray` function if your input
    has 3 channels (as the original MATLAB code does). Expected input image value range is [0, 255],
    and output quality scores are non-negative, where lower values indicate better perceptual quality.

    Warning: This metric is designed for natural images and may yield unreliable scores for non-natural images,
    as it relies on the assumption that the features extracted from the input image should follow a
    similar distribution to those extracted from natural images.

    Implementation based on the original MATLAB code using their extracted parameters:
    http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Original publication:
    Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik.
    "Making a “completely blind” image quality analyzer."
    IEEE Signal processing letters 20.3 (2012): 209-212.

    """

    @property
    def name(self) -> str:
        return "Natural Image Quality Evaluator"

    @property
    def abbreviation(self) -> str:
        return "NIQE"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self,
        block_size_row: int = 96,
        block_size_column: int = 96,
        block_column_overlap: int = 0,
        block_row_overlap: int = 0,
    ) -> None:
        """
        :param block_size_row: The patch size in the row dimension. Must be a positive integer. Default is 96.
        :type block_size_row: int
        :param block_size_column: The patch size in the column dimension. Must be a positive integer. Default is 96.
        :type block_size_column: int
        :param block_column_overlap: The overlap between adjacent patches in the column dimension.
        Must be a non-negative integer less than the corresponding block size. Default is 0.
        :type block_column_overlap: int
        :param block_row_overlap: The overlap between adjacent patches in the row dimension.
        Must be a non-negative integer less than the corresponding block size. Default is 0.
        :type block_row_overlap: int
        """
        super().__init__()
        self.block_size_row = block_size_row
        self.block_size_column = block_size_column
        self.block_column_overlap = block_column_overlap
        self.block_row_overlap = block_row_overlap

        if self.block_size_row <= 0:
            raise ValueError(f"block size row must be positive, but got {self.block_size_row}.")
        if self.block_size_column <= 0:
            raise ValueError(f"block size column must be positive, but got {self.block_size_column}.")
        if self.block_column_overlap < 0:
            raise ValueError(f"block column overlap must be non-negative, but got {self.block_column_overlap}.")
        if self.block_row_overlap < 0:
            raise ValueError(f"block row overlap must be non-negative, but got {self.block_row_overlap}.")

        try:
            PACKAGEDIR = Path(__file__).parent.absolute()
            model_parameters = loadmat(PACKAGEDIR / "niqe.mat")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "NIQE model parameters file not found. Please ensure that 'niqe.mat' is located next to this file."
            ) from e

        self.mu_prisparam = torch.from_numpy(model_parameters["mu_prisparam"]).double()
        self.cov_prisparam = torch.from_numpy(model_parameters["cov_prisparam"]).double()

        if self.mu_prisparam.shape != (1, 36) or self.cov_prisparam.shape != (36, 36):
            raise ValueError(
                f"NIQE model parameters have incorrect shapes. Expected mu_prisparam to have shape (36,) and "
                f"cov_prisparam to have shape (36, 36), but got {self.mu_prisparam.shape} and "
                f"{self.cov_prisparam.shape}."
            )

    def _compute(self, image: torch.Tensor) -> torch.Tensor:
        self._input_checks(image)

        # Crop the image so that it can be divided into non-overlapping blocks of the specified size
        number_block_rows = image.shape[-2] // self.block_size_row
        number_block_columns = image.shape[-1] // self.block_size_column
        image = image[..., : number_block_rows * self.block_size_row, : number_block_columns * self.block_size_column]

        number_block_rows = image.shape[-2] // self.block_size_row
        number_block_columns = image.shape[-1] // self.block_size_column
        image = image[..., : number_block_rows * self.block_size_row, : number_block_columns * self.block_size_column]

        window = gaussian_filter_kernel(7, sigma=7 / 6, device=image.device, dtype=image.dtype)
        window = window[3, :] / torch.sum(window[3, :])
        number_features = 18
        number_scales = 2

        features = image.new_zeros(number_block_columns * number_block_rows, number_scales * number_features)

        for i in range(1, number_scales + 1):
            # compute mean subtracted contrast normalized image
            mu = self.correlate1d(image, window, axis=1)
            mu = self.correlate1d(mu, window, axis=0)
            mu_squared = torch.pow(mu, 2)
            image_squared = torch.pow(image, 2)
            sigma = self.correlate1d(image_squared, window, axis=1)
            sigma = self.correlate1d(sigma, window, axis=0)
            sigma = torch.sqrt(torch.abs(sigma - mu_squared))
            mean_subtracted_contrast_normalized_image = (image - mu) / (sigma + 1.0)

            if not self._is_normal(mean_subtracted_contrast_normalized_image):
                logger.warning(
                    "The pixel values of the mean subtracted contrast normalized image do not appear to be"
                    "normally distributed. This may lead to unreliable NIQE scores, as NIQE relies on the "
                    "assumption of normality in the features extracted from natural images. Please ensure "
                    "that your input images are natural and that the preprocessing steps are correctly applied."
                )

            # feature extraction
            features_scale = self._block_process(
                mean_subtracted_contrast_normalized_image,
                [self.block_size_row // i, self.block_size_column // i],
                [self.block_row_overlap // i, self.block_column_overlap // i],
                self._compute_features,
            )
            features[:, (i - 1) * number_features : i * number_features] = features_scale

            # downsampling similar to MATLAB's imresize with bicubic interpolation and anti-aliasing
            image_padded = (
                torch.nn.functional.pad(image.unsqueeze(0).unsqueeze(0), pad=[2, 2, 2, 2], mode="reflect")
                .squeeze(0)
                .squeeze(0)
            )
            image = (
                torch.nn.functional.interpolate(
                    image_padded.unsqueeze(0).unsqueeze(0),
                    scale_factor=0.5,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            image = image[..., 1:-1, 1:-1]

        # Fit multivariate Gaussian to distorted patch features
        features = features.T
        mu_distparam = torch.nanmean(features, dim=1)
        cov_distparam = torch.cov(features)

        # compute quality score as Mahalanobis distance between fitted distribution and natural image distribution
        invcov_param = torch.linalg.pinv((self.cov_prisparam.to(image.device) + cov_distparam) / 2)
        return torch.sqrt(
            (self.mu_prisparam.to(image.device) - mu_distparam)
            @ invcov_param
            @ (self.mu_prisparam.to(image.device) - mu_distparam).T
        )

    def correlate1d(self, input: torch.Tensor, kernel: torch.Tensor, axis: int) -> torch.Tensor:
        """Perform 1D correlation of the input image with the given kernel along the
        specified axis. This function uses 2D convolution to perform the correlation by
        reshaping the kernel appropriately and applying padding to the input image.

        This function is designed to replicate the behavior of MATLAB's `imfilter` function with replicate padding.

        :param input: The input image to be correlated, expected to have shape (H, W).
        :type input: torch.Tensor
        :param kernel: The 1D kernel to be used for correlation, expected to have shape (7,).
        :type kernel: torch.Tensor
        :param axis: The axis along which to perform the correlation.
          Must be either 0 (for row-wise correlation) or 1 (for column-wise correlation).
        :type axis: int
        :return: The result of the 1D correlation, with the same shape as the input image.
        :rtype: torch.Tensor
        :raises ValueError: If the axis parameter is not 0 or 1, or if the kernel does not have the expected shape.

        """
        if axis == 0:
            kernel = kernel.view(7, 1)
            padding = [0, 0, 3, 3]
        elif axis == 1:
            kernel = kernel.view(1, 7)
            padding = [3, 3, 0, 0]
        else:
            raise ValueError(f"Axis must be 0 or 1, but got {axis}.")

        input_padded = (
            torch.nn.functional.pad(input.unsqueeze(0).unsqueeze(0), pad=padding, mode="replicate")
            .squeeze(0)
            .squeeze(0)
        )
        return convolve2d(input_padded, kernel, padding="valid")

    def _block_process(
        self,
        image: torch.Tensor,
        block_size: list[int],
        block_overlap: list[int],
        function: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply a function to blocks/patches of the input image, with optional overlap
        between blocks.

        :param image: The input image to be processed, expected to have shape (H, W).
        :type image: torch.Tensor
        :param block_size: The size of the blocks/patches to process, specified as
            [block_size_row, block_size_column]. Must be a list of two positive
            integers.
        :type block_size: list[int]
        :param block_overlap: The overlap between adjacent blocks/patches, specified as
            [block_row_overlap, block_column_overlap]. Must be a list of two non-
            negative integers less than the corresponding block size.
        :type block_overlap: list[int]
        :param function: The function to apply to each block/patch. This function
            should take a block of the image as input and return a tensor of features.
            The block passed to the function will have shape (block_size_row,
            block_size_column).
        :type function: Callable[[torch.Tensor], torch.Tensor]
        :return: A tensor containing the features extracted from each block/patch, with
            shape (num_blocks, feature_dim), where num_blocks is the total number of
            blocks processed and feature_dim is the dimensionality of the features
            returned by the function.
        :rtype: torch.Tensor

        """
        step_row = block_size[0] - block_overlap[0]
        step_col = block_size[1] - block_overlap[1]

        features = []
        for row in range(0, image.shape[-2] - block_size[0] + 1, step_row):
            for col in range(0, image.shape[-1] - block_size[1] + 1, step_col):
                block_features = image[..., col : col + block_size[1], row : row + block_size[0]]
                features.append(function(block_features))

        return torch.stack(features)

    def _compute_features(self, patch: torch.Tensor) -> torch.Tensor:
        """Computes the NIQE features for a given patch of the image. This includes
        estimating the parameters of the asymmetric generalized Gaussian distribution
        (AGGD) for the patch itself and for the products of the patch with its shifted
        versions.

        :param patch: The input patch of the image for which to compute the features,
            expected to have shape (block_size_row, block_size_column).
        :type patch: torch.Tensor
        :return: A tensor containing the computed features for the input patch, with
            shape (feature_dim,), where feature_dim is the total number of features
            extracted from the patch. The features include the AGGD parameters for the
            patch and for the products of the patch with its shifted versions.
        :rtype: torch.Tensor

        """
        features = []
        alpha, betal, betar = self._estimate_aggd_parameters(patch)
        features.extend([alpha, (betal + betar) / 2])

        shifts = torch.tensor([[0, 1], [1, 0], [1, 1], [1, -1]])

        for itr_shift in range(4):
            shifted_structdis = torch.roll(patch, shifts[itr_shift].tolist(), dims=(0, 1))
            pair = patch.flatten() * shifted_structdis.flatten()
            alpha, betal, betar = self._estimate_aggd_parameters(pair)
            meanparam = (betar - betal) * (
                self._gamma(torch.tensor(2.0) / alpha) / self._gamma(torch.tensor(1.0) / alpha)
            )
            features.extend([alpha, meanparam, betal, betar])

        return torch.tensor(features)

    def _gamma(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to compute the gamma function using the logarithm of the
        gamma function for numerical stability.

        :param x: The input tensor for which to compute the gamma function. Expected to
            be a 1D tensor containing the values for which to compute the gamma
            function.
        :type x: torch.Tensor
        :return: A tensor containing the computed gamma function values for the input
            tensor, with the same shape as the input tensor.
        :rtype: torch.Tensor

        """
        return torch.exp(torch.special.gammaln(x))

    def _estimate_aggd_parameters(self, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate the parameters of the asymmetric generalized Gaussian distribution
        (AGGD) for a given vector of values. This function computes the shape parameter
        (alpha) and the left and right scale parameters (betal and betar) of the AGGD
        based on the input vector.

        :param vector: The input vector for which to estimate the AGGD parameters,
            expected to be a 1D tensor containing the values from which to estimate the
            parameters.
        :type vector: torch.Tensor
        :return: A tuple containing the estimated AGGD parameters: (alpha, betal,
            betar), where alpha is the shape parameter, betal is the left scale
            parameter, and betar is the right scale parameter. Each of these parameters
            is returned as a tensor.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        """

        gam = torch.arange(0.2, 10.001, 0.001, device=vector.device, dtype=vector.dtype)
        r_gam = (self._gamma(2.0 / gam) ** 2) / (self._gamma(1.0 / gam) * self._gamma(3.0 / gam))

        left_std = torch.sqrt(torch.mean(vector[vector < 0] ** 2))
        right_std = torch.sqrt(torch.mean(vector[vector > 0] ** 2))

        gamma_hat = left_std / right_std
        r_hat = (torch.mean(torch.abs(vector)) ** 2) / torch.mean(vector**2)
        r_hat_norm = r_hat * ((gamma_hat**3 + 1) * (gamma_hat + 1)) / ((gamma_hat**2 + 1) ** 2)
        alpha = gam[torch.argmin((r_gam - r_hat_norm) ** 2)]

        betal = left_std * torch.sqrt(self._gamma(1.0 / alpha) / self._gamma(3.0 / alpha))
        betar = right_std * torch.sqrt(self._gamma(1.0 / alpha) / self._gamma(3.0 / alpha))

        return alpha, betal, betar

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        """Override this method in subclasses to register other implementations of the
        metric for comparison. Use the `_register_implementation` helper method to add
        implementations to the internal dictionary. These implementations will be used
        when compare_implementations is True.

        Store reference implementations in the `third_party` submodule of the metrics module,
        and import them here to register them for comparison.

        """
        self._register_implementation(implementations, "medimetric", get_medimetrics_niqe())

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} with "
            f"block_size_row={self.block_size_row}, block_size_column={self.block_size_column}, "
            f"block_row_overlap={self.block_row_overlap}, block_column_overlap={self.block_column_overlap}"
        )

    def _input_checks(self, image: torch.Tensor) -> None:
        """Perform input checks specific to NIQE, such as checking for valid pixel
        value ranges and dimensions.

        Warns if all pixel values in image are in the range [0, 1],
        which may indicate that the images are not correctly scaled for NIQE, which
        expects pixel values in the range [0, 255].

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        raises ValueError: If the input image contains pixel values outside the range
            [0, 255].

        """
        # Input checks specific to NIQE
        check_value_range(image, 0, 255)
        warn_if_all_pixels_in_0_to_1_range(image, image)

        if self.block_size_row > image.shape[-2] or self.block_size_column > image.shape[-1]:
            raise ValueError(
                f"Block size cannot be larger than the corresponding image dimension. "
                f"Got block_size_row={self.block_size_row}, block_size_column={self.block_size_column}, "
                f"but image has shape {image.shape}."
            )

    def _is_normal(self, image: torch.Tensor) -> bool:
        """Check if the pixel values of the image follow a normal distribution using
        skewness, kurtosis, and the Shapiro-Wilk test.

        :param image: The input image for which to check normality, expected to have
            shape (H, W).
        :type image: torch.Tensor
        :return: A boolean indicating whether the pixel values of the image are
            approximately normally distributed.
        :rtype: bool

        """
        pixel_values = image.flatten().cpu().numpy()

        if len(pixel_values) > 5000:
            rng = numpy.random.default_rng(42)
            pixel_values = rng.choice(pixel_values, size=5000, replace=False)

        skewness = stats.skew(pixel_values)
        kurtosis = stats.kurtosis(pixel_values)
        _, shapiro_p_value = stats.shapiro(pixel_values)
        return bool(abs(skewness) < 0.5 and abs(kurtosis) < 1.0 and shapiro_p_value > 0.05)
