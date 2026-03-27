from typing import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.utils.conversions import rgb_to_yiq
from mondAI.utils.similarity_map import similarity_map

logger = get_logger()


class HaarPSI(FullReferenceMetric):
    """Haar wavelet-based perceptual similarity index (HaarPSI) computes the perceptual
    similarity between two images based on features from their Haar wavelet
    coefficients. It is designed to capture perceptual differences between images which
    align with human visual perception by optimizing the parameters C and alpha. While
    for natural images C=30 and alpha=4.2 are recommended, for medical images C=5 and
    alpha=4.9 are recommended, as shown in Karner et al. (2025), which are the defaults
    in HaarPSI_MED.

    It expects two-dimensional grayscale or RGB (set `use_rgb` to True) images with pixel values in the range [0, 255].
    The resulting HaarPSI score ranges from 0 to 1, where a score of 1 indicates perfect
    similarity between the input image and the reference image, while a score of 0 indicates
    no perceptual similarity. Note that the RGB definition is not just a simple channel-wise application of the
    grayscale definition,
    but rather a different definition that considers the Y, I and Q channels of the YIQ color space separately.

    HaarPSI uses the magnitudes of high-frequency Haar wavelet coefficients to compute local similarities and the
    magnitudes of low-frequency Haar wavelet coefficients to compute weights for these local similarities.
    For the mathematical formulation of the metric, please refer to
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
    )  # for RGB images when `use_rgb == True` this changes to (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self, preprocess_with_subsampling: bool = True, C: float = 30.0, alpha: float = 4.2, use_rgb: bool = False
    ) -> None:
        """Initialize HaarPSI metric with parameter settings recommended for natural
        images.

        :param preprocess_with_subsampling: Whether to preprocess the images with
            subsampling to accommodate for viewing distance in psychophysical
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
        :param use_rgb: Whether to use metric definition for RGB images instead of
            grayscale definition. Note that the RGB definition is different from
            applying the grayscale definition to each channel separately and then
            aggregate. If True, the metric will expect 3-channel RGB images. Default is
            False (grayscale).
        :type use_rgb: bool

        """

        super().__init__()
        self.preprocess_with_subsampling = preprocess_with_subsampling
        self.C = C
        self.alpha = alpha
        self.use_rgb = use_rgb
        if self.use_rgb:
            self.expected_dimensions = (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)  # type: ignore[assignment]

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

        logger.info(
            "HaarPSI_MED was selected with parameters C=5.0 and alpha=4.9, which are recommended for medical images "
            "based on the publication by Karner et al. (2025). If you intended to use the parameter settings "
            "recommended for natural images, please use HaarPSI with C=30.0 and alpha=4.2."
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

        # Convert from RGB to YIQ color space
        if self.use_rgb:
            image = rgb_to_yiq(image)
            reference = rgb_to_yiq(reference)

        # Downscale input to simulates the typical distance between an image and its viewer.
        if self.preprocess_with_subsampling:
            image = self._subsample(image)
            reference = self._subsample(reference)

        # Perform Haar wavelet decomposition on 3 scales
        n_scales = 3
        coefficients_reference = self._haar_wavelet_decomposition(reference[0] if self.use_rgb else reference, n_scales)
        coefficients_image = self._haar_wavelet_decomposition(image[0] if self.use_rgb else image, n_scales)

        # Pre-allocate variables for the local similarities and the weights
        n_orientations = 2  # consider vertical and horizontal orientations of a 2D discrete Haar wavelet transform
        n_feature_channels = n_orientations + 1 if self.use_rgb else n_orientations  # grayscale: 2 , RGB 3 (Y, I, Q)
        local_similarities = reference.new_zeros((n_feature_channels, *reference.shape))  # (2 or 3, H, W)
        weights = reference.new_zeros(local_similarities.shape)  # (2 or 3, H, W)

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

        # Compute similarities for color channels (third feature channel)
        if self.use_rgb:

            def magnitude(image: torch.Tensor) -> torch.Tensor:
                return torch.abs(self._convolve2d(image, image.new_ones((2, 2)) / 4.0))

            coefficients_reference_I = magnitude(reference[1])
            coefficients_image_I = magnitude(image[1])
            coefficients_reference_Q = magnitude(reference[2])
            coefficients_image_Q = magnitude(image[2])

            similarity_I = similarity_map(coefficients_reference_I, coefficients_image_I, self.C)
            similarity_Q = similarity_map(coefficients_reference_Q, coefficients_image_Q, self.C)
            weights[2] = (weights[0] + weights[1]) / 2
            local_similarities[2] = (similarity_I + similarity_Q) / 2

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

        ### PIQ ###
        try:
            from piq import haarpsi

            def piq_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # piq's implementation expects inputs with shape (N, C, H, W) and
                # data_range parameter should correspond to pixel value range

                if not self.use_rgb:
                    image = image.unsqueeze(0)
                    reference = reference.unsqueeze(0)

                return haarpsi(
                    image.unsqueeze(0),
                    reference.unsqueeze(0),
                    data_range=255.0,
                    c=self.C,
                    alpha=self.alpha,
                    subsample=self.preprocess_with_subsampling,
                )

            implementations["piq"] = piq_haarpsi

        except Exception:
            logger.warning("piq or its HaarPSI implementation is not available, skipping piq implementation of HaarPSI")

        ### PIQA ###

        try:
            from piqa.haarpsi import haarpsi as haarpsi_piqa

            def piqa_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # piqa's implementation expects inputs with shape (N, C, H, W) and
                # data_range parameter should correspond to pixel value range

                if not self.use_rgb:
                    image = image.unsqueeze(0)
                    reference = reference.unsqueeze(0)

                return haarpsi_piqa(
                    image.unsqueeze(0).float() / 255.0,
                    reference.unsqueeze(0).float() / 255.0,
                    value_range=1.0,
                    c=self.C,
                    alpha=self.alpha,
                )

            implementations["piqa"] = piqa_haarpsi
        except Exception:
            logger.warning(
                "piqa or its HaarPSI implementation is not available, skipping piqa implementation of HaarPSI"
            )

        ### IdealIQA ###

        try:
            from haarpsi_ideal_iqa import haarpsi as haarpsi_ideal

            def ideal_iqa_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # ideal_iqa's implementation expects inputs with shape (N, C, H, W) and
                # pixel values in [0, 1], while the original implementation expects images with pixel values in [0, 255]

                score, _, _ = haarpsi_ideal(
                    reference / 255.0,
                    image / 255.0,
                    C=self.C,
                    α=self.alpha,
                    preprocess_with_subsampling=self.preprocess_with_subsampling,
                )
                return score

            implementations["ideal_iqa"] = ideal_iqa_haarpsi

        except Exception:
            logger.warning(
                "haarpsi_ideal_iqa or its HaarPSI implementation is not available, "
                "skipping ideal_iqa implementation of HaarPSI"
            )

        ### deepinv ###
        try:
            from deepinv.loss.metric import HaarPSI as DeepInvHaarPSI

            def deepinv_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # DeepInv's implementation expects inputs with shape (N, C, H, W) and
                #  pixel values in [0, 1] with single precision (float32, therefore scores might deviate slightly)

                if not self.use_rgb:
                    image = image.unsqueeze(0)
                    reference = reference.unsqueeze(0)

                haarpsi_metric = DeepInvHaarPSI(
                    C=self.C, alpha=self.alpha, preprocess_with_subsampling=self.preprocess_with_subsampling
                )
                return haarpsi_metric(
                    image.unsqueeze(0).float() / 255.0,
                    reference.unsqueeze(0).float() / 255.0,
                )

            implementations["deepinv"] = deepinv_haarpsi

        except Exception:
            logger.warning(
                "deepinv or its HaarPSI implementation is not available, skipping deepinv implementation of HaarPSI"
            )

        ### Original NumPy implementation by Rafael Reisenhofer and David Neumann ###

        try:
            from haarpsi_original import haar_psi as org_haarpsi

            def original_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # The original implementation by Rafael Reisenhofer and David Neumann expects images with pixel values
                # in [0, 255] and uses double precision floating point format.

                score, _, _ = org_haarpsi(
                    reference.numpy(force=True),
                    image.numpy(force=True),
                    preprocess_with_subsampling=self.preprocess_with_subsampling,
                )
                return torch.tensor(score)

            implementations["original_numpy"] = original_haarpsi
        except Exception:
            logger.warning(
                "haarpsi_original or its HaarPSI implementation is not available, "
                "skipping original_numpy implementation of HaarPSI"
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
        rgb_info = " using RGB definition" if self.use_rgb else ""
        return (
            f"{self.name} ({self.abbreviation}) {arrow} with C={self.C} and alpha={self.alpha}{preprocessing}{rgb_info}"
        )

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
            "use_rgb": self.use_rgb,
        }

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to HaarPSI, such as checking for valid pixel
        value ranges and dimensions.

        Warns if the images have height or width smaller than 16 pixels, which may lead
        to unreliable results due to border effects of the 16x16 kernel used in
        HaarPSI. Warns if all pixel values in both image and reference are in the range
        [0, 1], which may indicate that the images are not correctly scaled for
        HaarPSI, which expects pixel values in the range [0, 255].

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :raises ValueError: If the input image contains pixel values outside the range
            [0, 255].
        :raises ValueError: If the reference image contains pixel values outside the
            range [0, 255].
        :raises ValueError: If use_rgb is True but the images do not have 3 channels,
            which is required for the RGB definition of HaarPSI.

        """
        # Input checks specific to HaarPSI
        if torch.any(image < 0) or torch.any(image > 255):
            raise ValueError("Input image contains pixel values outside the range [0, 255].")

        if torch.any(reference < 0) or torch.any(reference > 255):
            raise ValueError("Reference image contains pixel values outside the range [0, 255].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            logger.warning(
                "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "HaarPSI expects pixel values in the range [0, 255]. Please ensure that your input images are "
                "correctly scaled for accurate computation of HaarPSI."
            )

        if any(
            [reference.shape[self.expected_dimensions.index(dim)] < 16 for dim in (Dimension.HEIGHT, Dimension.WIDTH)]
        ):
            logger.warning(
                "Input image has height or width smaller than 16 pixels. HaarPSI uses a 16x16 kernel, "
                "so results may be unreliable for small images due to border effects."
            )

        if any([image.shape[self.expected_dimensions.index(dim)] < 16 for dim in (Dimension.HEIGHT, Dimension.WIDTH)]):
            logger.warning(
                "Images have height or width smaller than 16 pixels. HaarPSI uses a 16x16 kernel, "
                "so results may be unreliable for small images due to border effects."
            )

        if self.use_rgb:
            channel_dim = self.expected_dimensions.index(Dimension.CHANNEL)
            if image.shape[channel_dim] != 3:
                raise ValueError(
                    f"Images have {image.shape[channel_dim]} channels. HaarPSI expects 3 (RGB) channels, "
                    "when use_rgb is set to True. Instead you might want to use the grayscale definition"
                    "(use_rgb=False) and apply it channel wise."
                )

    def _subsample(self, image: torch.Tensor) -> torch.Tensor:
        """Subsample the input image by a factor of 2 using a 2x2 mean filter and
        dyadic subsampling. This simulates the typical distance between an image and
        its viewer in psychophysical experiments as described in the original
        publication.

        If use_rgb is True and the input image has 3 channels, the subsampling is
        applied to each channel separately and then the subsampled channels are stacked
        back together.

        :param image: The input 2D image to be subsampled, shape (H, W), or (C, H, W)
            if use_rgb is True and the image has 3 channels.
        :type image: torch.Tensor
        :return: The subsampled image, shape (H/2, W/2), or (H/2+1, W/2+1) if the input
            dimensions are odd. If use_rgb is True and the input image has 3 channels,
            the output shape will be (C, H/2, W/2) or (C, H/2+1, W/2+1) if the input
            dimensions are odd.
        :rtype: torch.Tensor

        """

        # apply subsampling to each channel separately and stack back together
        if self.use_rgb and image.shape[self.expected_dimensions.index(Dimension.CHANNEL)] == 3:
            subsampled_channels = [self._subsample(image[channel]) for channel in range(3)]
            return torch.stack(subsampled_channels, dim=self.expected_dimensions.index(Dimension.CHANNEL))

        kernel_size = 2
        filter_weights = image.new_ones(1, 1, kernel_size, kernel_size) / kernel_size**2
        mean_filtered = torch.nn.functional.conv2d(image.unsqueeze(0), weight=filter_weights, padding="same")
        subsampled = mean_filtered.squeeze()[::kernel_size, ::kernel_size]
        return subsampled

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


class HaarPSI_MED(HaarPSI):
    """Haar wavelet-based perceptual similarity index (HaarPSI) with parameter settings
    recommended for medical images based on the publication by Karner et al (2025)."""

    def __init__(
        self, preprocess_with_subsampling: bool = True, C: float = 5.0, alpha: float = 4.9, use_rgb: bool = False
    ) -> None:
        """Initialize HaarPSI with parameter settings recommended for medical images
        based on the publication by Karner et al. (2025).

        :param preprocess_with_subsampling: Whether to preprocess the images with
            subsampling to accommodate for viewing distance in psychophysical
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
        :param use_rgb: Whether to use metric definition for RGB images instead of
            grayscale definition. Note that the RGB definition is different from
            applying the grayscale definition to each channel separately and then
            aggregate. If True, the metric will expect 3-channel RGB images. Default is
            False (grayscale).
        :type use_rgb: bool

        """
        super().__init__(preprocess_with_subsampling=preprocess_with_subsampling, C=C, alpha=alpha, use_rgb=use_rgb)
        logger.info(
            "HaarPSI_MED was selected with parameters C=5.0 and alpha=4.9, which are recommended for medical images "
            "based on the publication by Karner et al. (2025). If you intended to use the parameter settings "
            "recommended for natural images, please use HaarPSI with C=30.0 and alpha=4.2."
        )
