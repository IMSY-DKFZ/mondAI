from typing import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.utils.similarity_map import similarity_map

# from mondAI.utils.conversions import rgb_to_yiq

logger = get_logger()


class HaarPSI(FullReferenceMetric):
    """Haar wavelet-based perceptual similarity index (HaarPSI) computes the perceptual
    similarity between two images based on features from their Haar wavelet transform.
    It is designed to capture perceptual differences between images which align with
    human visual perception.

    It expects two-dimensional grayscale images with pixel values in the range [0, 255].
    The resulting HaarPSI score ranges from 0 to 1, where a score of 1 indicates perfect
    similarity between the input image and the reference image, while a score of 0 indicates
    no perceptual similarity. For the mathematical formulation of the metric, please refer to
    the original publication


    Implementation adapted from Anna Breger and Clemens Karner
    (https://github.com/ideal-iqa/haarpsi-pytorch/blob/main/haarpsi.py commit:
    a64b753b1b95a826996fcc035ce3f4dc4c630a5f, License: MIT), with the difference
    that this implementation expects images with pixel values in the range [0, 255]
    as the original publication, while their implementation expects images with
    pixel values in the range [0, 1]. Furthermore, this implementation uses a double
    precision floating point format for all computations, while their implementation
    uses single precision. The original MATLAB and NumPy implementations by Rafael
    Reisenhofer (https://github.com/rgcda/haarpsi/blob/master/HaarPSI.m commit:
    2c2793108477deb81971658a7666d5f85ba2587b, License: MIT) and David Neumann
    (https://github.com/rgcda/haarpsi/blob/master/haarPsi.py commit:
    2c2793108477deb81971658a7666d5f85ba2587b, License: MIT) also expect images with
    pixel values in the range [0, 255] and use double precision floating point
    format.

    Original publication:
    R. Reisenhofer, S. Bosse, G. Kutyniok and T. Wiegand.
    A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment.
    Signal Processing: Image Communication, vol. 61, 33-43, 2018.
    doi:10.1016/j.image.2017.11.001

    Parameter configuration for medical images based on this publication:
    Karner, C., Gröhl, J., Selby, I., Babar, J., Beckford, J., Else, T. R., Sadler, T. J.,
    Shahipasand, S., Thavakumar, A., Roberts, M., Rudd, J. H. F., Schönlieb, C.-B.,
    Weir-McCall, J. R., & Breger, A. (2025). Parameter choices in HaarPSI for IQA with
    medical images. 2025 IEEE International Symposium on Biomedical Imaging (ISBI).

    """

    name = "Haar wavelet-based perceptual similarity index"
    abbreviation = "HaarPSI"
    higher_is_better = True

    expected_dimensions = (
        Dimension.HEIGHT,
        Dimension.WIDTH,
    )  # for color images use HaarPSI_RGB instead, which expects (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, preprocess_with_subsampling: bool = True, C: float = 5.0, alpha: float = 4.9) -> None:
        """Initialize the Metric Template.

        The constructor should include checks for any parameters that the metric uses
        to ensure they are within valid ranges or meet certain conditions. For example,
        if the metric has a parameter that must be non-negative, this method should
        check that condition and raise a ValueError if it is not met. This helps to
        prevent invalid configurations of the metric that could lead to incorrect
        results or errors during computation.

        :param preprocess_with_subsampling: Whether to preprocess the images with
            subsampling to accomodate for viewing distance in psychophysical
            experiments as described in the original publication. Default is True
        :type preprocess_with_subsampling: bool
        :param C: A positive constant used in the computation of HaarPSI to avoid
            instability when the local similarity is close to zero. Default for medical
            images is 5.0, for natural images 30.0. The authors suggest to set it in
            the range [5.0, 100.0].
        :type C: float
        :param alpha: A positive constant used in the computation of HaarPSI to control
            the logistic function of the local similarity map. Default for medical
            images is 4.9, for natural images 4.2. The authors suggest to set it in the
            range [2.0, 8.0].
        :type alpha: float

        """

        super().__init__()
        self.preprocess_with_subsampling = preprocess_with_subsampling
        self.C = C
        self.alpha = alpha

        # Check parameter settings for validity
        if self.C <= 0:
            raise ValueError("C must be a positive float.")

        if not isinstance(self.C, float):
            if isinstance(self.C, int):
                self.C = float(self.C)
            else:
                raise ValueError("C must be a float.")

        if self.alpha <= 0:
            raise ValueError("alpha must be a positive float.")

        if not isinstance(self.alpha, float):
            raise ValueError("alpha must be a float.")

        # Warnings for parameter choices outside of recommended ranges, but still valid
        if not 5 <= self.C <= 100:
            logger.warning(
                "C should be set in the range [5, 100]. Please ensure that your choice of C "
                "is appropriate for your use case."
            )

        if not 2 <= self.alpha <= 8:
            logger.warning(
                "alpha should be set in the range [2, 8]. Please ensure that your choice of alpha "
                "is appropriate for your use case."
            )

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the metric between image and reference.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :return: The computed metric score.
        :rtype: torch.Tensor

        """

        self._input_checks(image, reference)

        # Convert to double precision if not already (as in original implementation)
        image = image.double()
        reference = reference.double()

        # Downscale input to simulates the typical distance between an image and its viewer.
        if self.preprocess_with_subsampling:
            image = self._subsample(image)
            reference = self._subsample(reference)

        # Perform Haar wavelet decomposition on 3 scales
        n_scales = 3
        coefficients_reference = self._haar_wavelet_decomposition(reference, n_scales)
        coefficients_image = self._haar_wavelet_decomposition(image, n_scales)

        # Pre-allocate variables for the local similarities and the weights
        n_orientations = 2  # consider vertical and horizontal orientations of a 2D discrete Haar wavelet transform
        local_similarities = reference.new_zeros(n_orientations, *reference.shape)  # (2, H, W)
        weights = reference.new_zeros(n_orientations, *reference.shape)  # (2, H, W)

        # Computes the weights and similarities for each orientation
        for orientation in range(n_orientations):
            # Low-frequency coefficients used as weights
            weights[orientation] = self._get_weights_for_orientation(
                coefficients_image, coefficients_reference, n_scales, orientation
            )

            # High-frequency coefficients used for local similarity
            local_similarities[orientation] = self._get_local_similarity_for_orientation(
                coefficients_image, coefficients_reference, n_scales, orientation
            )

        # Calculates the final score
        pre_logit = torch.sum(torch.sigmoid(self.alpha * local_similarities) * weights) / torch.sum(weights)
        similarity = (torch.log(pre_logit / (1 - pre_logit)) / self.alpha) ** 2

        return similarity  # , local_similarities, weights

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the metric. This will be
        used when compare_implementations is True to compute the metric using different
        libraries or implementations for comparison.

        :return: A dictionary where the keys are the names of the libraries or implementations, and the values are
        callables that compute the metric using those implementations.
        :rtype: dict[str, callable[..., torch.Tensor]]

        """

        implementations = {}

        ### PIG ###
        try:
            from piq import haarpsi

            def piq_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # piq's implementation expects inputs with shape (N, C, H, W) and
                # data_range parameter should correspond to pixel value range
                return haarpsi(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    data_range=255.0,
                    c=self.C,
                    alpha=self.alpha,
                )

            implementations["piq"] = piq_haarpsi

        except Exception:
            logger.warning(
                "piq or it's HaarPSI implementation is not available, skipping piq implementation of HaarPSI"
            )

        ### deepinv ###
        try:
            from deepinv.loss.metric import HaarPSI as DeepInvHaarPSI

            def deepinv_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # DeepInv's implementation expects inputs with shape (N, C, H, W) and
                #  pixel values in [0, 1] with single precision (float32, therefore scores might deviate slightly)
                haarpsi_metric = DeepInvHaarPSI(C=self.C, alpha=self.alpha)
                return haarpsi_metric(
                    image.unsqueeze(0).unsqueeze(0).float() / 255.0,
                    reference.unsqueeze(0).unsqueeze(0).float() / 255.0,
                )

            implementations["deepinv"] = deepinv_haarpsi

        except Exception:
            logger.warning(
                "deepinv or it's HaarPSI implementation is not available, skipping deepinv implementation of HaarPSI"
            )

        return implementations

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        arrow = self._arrow_indicating_optimum()
        preprocessing = " and subsampling as preprocessing" if self.preprocess_with_subsampling else ""
        return f"{self.name} ({self.abbreviation}) {arrow} with C={self.C} and alpha={self.alpha}{preprocessing}"

    def fingerprint(self) -> dict[str, str | bool | float]:
        """Return a dictionary that uniquely identifies the metric.

        :return: A dictionary that uniquely identifies the metric and its parameters
            for reproducibility.
        :rtype: dict[str, str | bool | float]

        """
        return {
            "name": self.name,
            "abbreviation": self.abbreviation,
            "higher_is_better": self.higher_is_better,
            "C": self.C,
            "alpha": self.alpha,
            "preprocess_with_subsampling": self.preprocess_with_subsampling,
        }

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """_summary_

        :param image: _description_
        :type image: torch.Tensor
        :param reference: _description_
        :type reference: torch.Tensor
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_

        """
        # Input checks specific to HaarPSI
        if torch.any(image < 0) or torch.any(image > 255):
            raise ValueError("Input image contains pixel values outside the range [0, 255].")

        if torch.any(reference < 0) or torch.any(reference > 255):
            raise ValueError("Reference image contains pixel values outside the range [0, 255].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            logger.warning(
                "It hase been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "HaarPSI expects pixel values in the range [0, 255]. Please ensure that your input images are "
                "correctly scaled for accurate computation of HaarPSI."
            )

        # # Check that only 1 or 3 channels are present
        # if image.shape[self.expected_dimensions.index(Dimension.CHANNEL)] not in (1, 3):
        #     raise ValueError(
        #         f"Input image has {image.shape[0]} channels. HaarPSI expects 1 (grayscale) or 3 (RGB) channels."
        #     )

        # TODO: check kernel size (based on n_scales = 3?) makes sense for image sizes.

    def _subsample(self, image: torch.Tensor) -> torch.Tensor:
        """Subsample the input image by a factor of 2 using a 2x2 mean filter and
        dyadic subsampling. This simulates the typical distance between an image and
        its viewer in psychophysical experiments as described in the original
        publication.

        :param image: The input 2D image to be subsampled, shape (H, W)
        :type image: torch.Tensor
        :return: The subsampled image, shape (H/2, W/2), or (H/2+1, W/2+1) if the input
            dimensions are odd.
        :rtype: torch.Tensor

        """
        kernel_size = 2
        filter_weights = image.new_ones(1, 1, kernel_size, kernel_size) / kernel_size**2
        mean_filtered = torch.nn.functional.conv2d(image.unsqueeze(0), weight=filter_weights, padding="same")
        subsumpled = mean_filtered.squeeze()[::kernel_size, ::kernel_size]
        return subsumpled

    def _convolve2d(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Convolve the input image with the given kernel using 2D convolution.

        :param image: The input 2D image to be convolved, shape (H, W)
        :type image: torch.Tensor
        :param kernel: The 2D convolution kernel
        :type kernel: torch.Tensor
        :return: The convolved image, shape (H, W)
        :rtype: torch.Tensor

        """
        convolved = torch.nn.functional.conv2d(
            image.unsqueeze(0), weight=kernel.unsqueeze(0).unsqueeze(0), padding="same"
        )
        return convolved.squeeze()

    def _haar_wavelet_decomposition(self, image: torch.Tensor, n_scales: int) -> torch.Tensor:
        """Perform a 2D Haar wavelet decomposition of the input image up to the
        specified number of scales.

        :param image: The input 2D image to be decomposed, shape (H, W)
        :type image: torch.Tensor
        :param n_scales: The number of scales to decompose the image into
        :type n_scales: int
        :return: The Haar wavelet coefficients, shape (2 * n_scales, H, W)
        :rtype: torch.Tensor

        """
        coefficients = image.new_zeros(2 * n_scales, *image.shape)  # (2*n_scales, H, W)

        def _get_haar_filter(scale: int) -> torch.Tensor:
            """Get the 2D Haar wavelet filter for the specified scale.

            :param scale: The scale for which to get the Haar filter (1-based index)
            :type scale: int
            :return: The 2D Haar wavelet filter for the specified scale, shape
                  (2^scale, 2^scale)
            :rtype: torch.Tensor

            """
            haar_filter = 2**-scale * image.new_ones(2**scale, 2**scale)
            haar_filter[: haar_filter.shape[0] // 2, :] = -haar_filter[: haar_filter.shape[0] // 2, :]
            return haar_filter

        for scale in range(n_scales):
            haar_filter = _get_haar_filter(scale + 1)
            coefficients[scale] = self._convolve2d(image, haar_filter)
            coefficients[scale + n_scales] = self._convolve2d(image, haar_filter.t())
        return coefficients

    def _get_weights_for_orientation(
        self,
        coefficients_image: torch.Tensor,
        coefficients_reference: torch.Tensor,
        n_scales: int,
        orientation: int,
    ) -> torch.Tensor:
        """Calculate the weights for the specified orientation based on maximum
        magnitudes of Haar wavelet coefficients.

        :param coefficients_image: The Haar wavelet coefficients of the input image,
            shape (2*n_scales, H, W)
        :type coefficients_image: torch.Tensor
        :param coefficients_reference: The Haar wavelet coefficients of the reference
            image, shape (2*n_scales, H, W)
        :type coefficients_reference: torch.Tensor
        :param n_scales: The number of scales in the Haar wavelet decomposition
        :type n_scales: int
        :param orientation: The orientation for which to calculate the weights
        :type orientation: int
        :return: The weights for the specified orientation, shape (H, W)
        :rtype: torch.Tensor

        """
        maximum_magnitude = torch.maximum(
            coefficients_reference[len(coefficients_reference) // n_scales + orientation * n_scales].abs(),
            coefficients_image[len(coefficients_image) // n_scales + orientation * n_scales].abs(),
        )
        return maximum_magnitude

    def _get_local_similarity_for_orientation(
        self,
        coefficients_image: torch.Tensor,
        coefficients_reference: torch.Tensor,
        n_scales: int,
        orientation: int,
    ) -> torch.Tensor:
        """Calculate local similarity for the specified orientation based on magnitudes
        of Haar wavelet coefficients.

        :param coefficients_image: The Haar wavelet coefficients of the input image,
            shape (2*n_scales, H, W)
        :type coefficients_image: torch.Tensor
        :param coefficients_reference: The Haar wavelet coefficients of the reference
            image, shape (2*n_scales, H, W)
        :type coefficients_reference: torch.Tensor
        :param n_scales: The number of scales in the Haar wavelet decomposition
        :type n_scales: int
        :param orientation: The orientation for which to calculate the local similarity
        :type orientation: int
        :return: The local similarity for the specified orientation, shape (H, W)
        :rtype: torch.Tensor

        """

        coefficients_reference_magnitude = coefficients_reference.abs()[
            (orientation * n_scales, 1 + orientation * n_scales), :, :
        ]
        coefficients_image_magnitude = coefficients_image.abs()[
            (orientation * n_scales, 1 + orientation * n_scales), :, :
        ]

        similarity_maps = similarity_map(coefficients_reference_magnitude, coefficients_image_magnitude, self.C)
        local_similarity = (similarity_maps[0] + similarity_maps[1]) / 2

        return local_similarity
