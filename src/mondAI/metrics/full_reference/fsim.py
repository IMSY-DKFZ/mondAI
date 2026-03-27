from typing import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.utils.conversions import rgb_to_yiq
from mondAI.utils.similarity_map import similarity_map

logger = get_logger()


class FSIM(FullReferenceMetric):
    """Feature SIMilarity (FSIM) index computes the perceptual similarity between two
    images based on phase congruency and gradient magnitude similarity. It is designed
    to capture perceptual differences between images which align with human visual
    perception. It expects two-dimensional grayscale or RGB (set `use_rgb` to True)
    images with pixel values in the range [0, 255]. The resulting FSIM score usually
    ranges from 0 to 1, where a score of 1 indicates perfect similarity between the
    input image and the reference image, while a score of 0 indicates no perceptual
    similarity. Note that the RGB definition is not just a simple channel-wise
    application of the grayscale definition, but rather a different definition that
    considers the Y, I and Q channels of the YIQ color space separately.

    FSIM uses the phase congruency and gradient magnitude of image to compute local similarities, and the
    phase congruency is used to compute weights for these local similarities.
    For the mathematical formulation of the metric, please refer to the original publication.

    Implementation adapted from PIQ (https://github.com/photosynthesis-team/piq/blob/master/piq/fsim.py,
    commit: 213a46687ad99098f274784e61d92c5144a94a68, License: Apache 2.0), which is based on the original
    MATLAB implementation by Lin Zhang, Lei Zhang, Xuanqin Mou and David Zhang
    (https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/Files/FeatureSIM.m)

    Original publication:
    L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
    IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
    https://ieeexplore.ieee.org/document/5705575

    """

    name = "Feature SIMilarity Index"
    abbreviation = "FSIM"
    higher_is_better = True

    expected_dimensions = (
        Dimension.HEIGHT,
        Dimension.WIDTH,
    )  # for RGB images when `use_rgb == True` this chanes to (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self,
        use_rgb: bool = False,
        T1: float = 0.85,
        T2: float = 160.0,
        T3: float = 200.0,
        T4: float = 200.0,
        _lambda: float = 0.03,
        scales: int = 4,
        orientations: int = 4,
        minimal_wavelength: int = 6,
        scaling_factor: int = 2,
        sigma_f: float = 0.55,
        delta_theta: float = 1.2,
        noise_threshold_factor: float = 2.0,
        epsilon: float = 1e-4,
    ) -> None:
        """Initialize FSIM metric with parameter settings recommended for natural
        images.

        :param use_rgb: Whether to use metric definition for RGB images instead of
            grayscale definition. Note that the RGB definition is different from
            applying the grayscale definition to each channel separately and then
            aggregate. If True, the metric will expect 3-channel RGB images. Default is
            False (grayscale).
        :type use_rgb: bool
        :param T1: The constant used in the phase congruency similarity function to
            avoid instability when the denominator is close to zero, default is 0.85 as
            in original implementation
        :type T1: float
        :param T2: The constant used in the gradient magnitude similarity function to
            avoid instability when the denominator is close to zero, default is 160.0
            as in original implementation
        :type T2: float
        :param T3: The constant used in the similarity function for the I channel when
            using RGB images to avoid instability when the denominator is close to
            zero, default is 200.0 as in original implementation
        :type T3: float
        :param T4: The constant used in the similarity function for the Q channel when
            using RGB images to avoid instability when the denominator is close to
            zero, default is 200.0 as in original implementation
        :type T4: float
        :param _lambda: The exponent used to weight the chromatic similarity in the
            final FSIM score when using RGB images, default is 0.03 as in original
            implementation
        :type _lambda: float
        :param scales: Number of wavelet scales, default is 4 as in original
            implementation
        :type scales: int
        :param orientations: Number of filter orientations, default is 4 as in original
            implementation
        :type orientations: int
        :param minimal_wavelength: The wavelength of the smallest scale filter, default
            is 6 as in original implementation
        :type minimal_wavelength: int
        :param scaling_factor: The scaling factor between successive filters, default
            is 2 as in original implementation
        :type scaling_factor: int
        :param sigma_f: The ratio of the standard deviation of the Gaussian describing
            the log Gabor filter's transfer function in the frequency domain to the
            filter center frequency, default is 0.55 as in original implementation
        :type sigma_f: float
        :param delta_theta: The ratio of the angular interval between filter
            orientations to the standard deviation of the Gaussian describing the log
            Gabor filter's transfer function in the frequency domain, default is 1.2 as
            in original implementation
        :type delta_theta: float
        :param noise_threshold_factor: Number of standard deviations above the noise
            mean for the noise compensation threshold, default is 2.0 as in original
            implementation. Below this threshold the response is considered to be
            dominated by noise and is suppressed.
        :type noise_threshold_factor: float
        :param epsilon: A small constant to avoid division by zero, default is 1e-4 as
            in original implementation
        :type epsilon: float

        """

        super().__init__()

        self.use_rgb = use_rgb
        if self.use_rgb:
            self.expected_dimensions = (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)  # type: ignore[assignment]

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self._lambda = _lambda

        # Parameters for phase congruency computation
        self.scales = scales
        self.orientations = orientations
        self.minimal_wavelength = minimal_wavelength
        self.scaling_factor = scaling_factor
        self.sigma_f = sigma_f
        self.delta_theta = delta_theta
        self.noise_threshold_factor = noise_threshold_factor
        self.epsilon = epsilon

        # Check parameter settings for validity
        if self.T1 <= 0 or self.T2 <= 0 or self.T3 <= 0 or self.T4 <= 0:
            raise ValueError("T1, T2, T3 and T4 must be positive to avoid instability in similarity calculations.")
        if self._lambda < 0:
            raise ValueError(
                "Lambda must be non-negative as it is used as an exponent for weighting chromatic similarity."
            )
        if self.scales <= 0:
            raise ValueError("Number of scales must be positive.")
        if self.orientations <= 0:
            raise ValueError("Number of orientations must be positive.")
        if self.minimal_wavelength <= 0:
            raise ValueError("Minimal wavelength must be positive.")
        if self.scaling_factor <= 1:
            raise ValueError(
                "Scaling factor must be greater than 1 to ensure that filters are properly spaced in frequency domain."
            )
        if self.sigma_f <= 0:
            raise ValueError("Sigma_f must be positive to ensure a valid log Gabor filter shape.")
        if self.delta_theta <= 0:
            raise ValueError("Delta_theta must be positive to ensure a valid log Gabor filter shape.")
        if self.noise_threshold_factor <= 0:
            raise ValueError("Noise threshold factor must be positive to ensure proper noise compensation.")
        if self.epsilon <= 0 or self.epsilon >= 0.1:
            raise ValueError(
                "Epsilon must be positive to avoid division by zero and less than 0.1 for numerical stability."
            )

        # warnings for non-default parameter settings
        if self.T1 != 0.85 or self.T2 != 160.0 or self.T3 != 200.0 or self.T4 != 200.0:
            logger.warning(
                "You have set T1, T2, T3 or T4 to non-default values (default: 0.85, 160.0, 200.0, 200.0)."
                " Please ensure that these values are appropriate for your specific use case, as they can affect"
                " the stability and sensitivity of the FSIM metric."
            )
        if self._lambda != 0.03:
            logger.warning(
                "You have set lambda to a non-default value (default: 0.03). "
                "Please ensure that this value is appropriate for your specific use case, as it affects the weighting"
                " of chromatic similarity in the final FSIM score when using RGB images."
            )

        non_default_params = []
        if self.scales != 4:
            non_default_params.append(f"scales={self.scales} (default: 4)")
        if self.orientations != 4:
            non_default_params.append(f"orientations={self.orientations} (default: 4)")
        if self.minimal_wavelength != 6:
            non_default_params.append(f"minimal_wavelength={self.minimal_wavelength} (default: 6)")
        if self.scaling_factor != 2:
            non_default_params.append(f"scaling_factor={self.scaling_factor} (default: 2)")
        if self.sigma_f != 0.55:
            non_default_params.append(f"sigma_f={self.sigma_f} (default: 0.55)")
        if self.delta_theta != 1.2:
            non_default_params.append(f"delta_theta={self.delta_theta} (default: 1.2)")
        if self.noise_threshold_factor != 2.0:
            non_default_params.append(f"noise_threshold_factor={self.noise_threshold_factor} (default: 2.0)")
        if self.epsilon != 1e-4:
            non_default_params.append(f"epsilon={self.epsilon} (default: 1e-4)")

        if non_default_params:
            logger.warning(
                f"Non-default FSIM parameters detected: {', '.join(non_default_params)}. "
                "Please ensure these values are appropriate for your specific use case, as they may affect "
                "the stability, sensitivity, and accuracy of the FSIM metric."
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

        # Convert from RGB to YIQ color space
        if self.use_rgb:
            image = rgb_to_yiq(image)
            reference = rgb_to_yiq(reference)

        # Convert to double precision if not already (as in original implementation)
        image = image.double()
        reference = reference.double()

        # Downsample the images
        min_dimension = min(
            image.shape[self.expected_dimensions.index(dim)] for dim in (Dimension.HEIGHT, Dimension.WIDTH)
        )
        kernel_size = max(1, round(min_dimension / 256))
        image = self._subsample(image, kernel_size=kernel_size)
        reference = self._subsample(reference, kernel_size=kernel_size)

        # Compute phase congruency maps
        phase_congruency_image = self._phase_congruency(
            image[0] if self.use_rgb else image,
            self.scales,
            self.orientations,
            self.minimal_wavelength,
            self.scaling_factor,
            self.sigma_f,
            self.delta_theta,
            self.noise_threshold_factor,
            self.epsilon,
        )
        phase_congruency_reference = self._phase_congruency(
            reference[0] if self.use_rgb else reference,
            self.scales,
            self.orientations,
            self.minimal_wavelength,
            self.scaling_factor,
            self.sigma_f,
            self.delta_theta,
            self.noise_threshold_factor,
            self.epsilon,
        )

        # Compute gradient magnitude maps
        sharr_kernel = (
            torch.outer(
                torch.tensor([3.0, 10.0, 3.0]) / 16,
                torch.tensor([1.0, 0.0, -1.0]),
            )
            .double()
            .to(image.device)
        )

        def gradient_map(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
            gradient_x = self._convolve2d(image, kernel)
            gradient_y = self._convolve2d(image, kernel.t())
            return torch.sqrt(gradient_x**2 + gradient_y**2)

        gradient_map_image = gradient_map(image[0] if self.use_rgb else image, sharr_kernel)
        gradient_map_reference = gradient_map(reference[0] if self.use_rgb else reference, sharr_kernel)

        # Calculate similarity maps and weights, and then compute the final FSIM score
        pc_similarity = similarity_map(phase_congruency_image, phase_congruency_reference, self.T1)
        gradient_similarity = similarity_map(gradient_map_image, gradient_map_reference, self.T2)
        pc_maximum = torch.maximum(phase_congruency_image, phase_congruency_reference)

        score = torch.sum(pc_similarity * gradient_similarity * pc_maximum) / torch.sum(pc_maximum)

        if self.use_rgb:
            similarity_I = similarity_map(image[1], reference[1], self.T3)
            similarity_Q = similarity_map(image[2], reference[2], self.T4)

            score = torch.nansum(
                pc_similarity * gradient_similarity * pc_maximum * (similarity_I * similarity_Q).real ** self._lambda
            ) / torch.sum(pc_maximum)

        return score  # phase_congruency_image, phase_congruency_reference, gradient_map_image, gradient_map_reference

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
            from piq import fsim

            def piq_fsim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # piq's implementation expects inputs with shape (N, C, H, W) and
                # data_range parameter should correspond to pixel value range

                if not self.use_rgb:
                    image = image.unsqueeze(0)
                    reference = reference.unsqueeze(0)

                return fsim(
                    image.unsqueeze(0),
                    reference.unsqueeze(0),
                    data_range=255.0,
                    chromatic=self.use_rgb,
                    scales=self.scales,
                    orientations=self.orientations,
                    min_length=self.minimal_wavelength,
                    mult=self.scaling_factor,
                    sigma_f=self.sigma_f,
                    delta_theta=self.delta_theta,
                    k=self.noise_threshold_factor,
                )

            implementations["piq"] = piq_fsim

        except Exception:
            logger.warning("piq or it's FSIM implementation is not available, skipping piq implementation of FSIM")

        ### PIQA ###

        try:
            from piqa.fsim import fsim as fsim_piqa
            from piqa.fsim import gradient_kernel, pc_filters, phase_congruency, scharr_kernel

            def piqa_fsim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # piqa's implementation expects inputs with shape (N, C, H, W) and
                # data_range parameter should correspond to pixel value range

                image = image.unsqueeze(0)
                reference = reference.unsqueeze(0)

                if not self.use_rgb:
                    image = image.unsqueeze(0)
                    reference = reference.unsqueeze(0)

                filters_1 = pc_filters(image)
                filters_2 = pc_filters(reference)
                pc_1 = phase_congruency(image[:, :1, :, :] if image.shape[1] == 3 else image, filters_1)
                pc_2 = phase_congruency(reference[:, :1, :, :] if reference.shape[1] == 3 else reference, filters_2)
                kernel = gradient_kernel(scharr_kernel().to(image.device))

                return fsim_piqa(
                    image.float() * 255.0,
                    reference.float() * 255.0,
                    pc_1,
                    pc_2,
                    kernel,
                    value_range=255.0,
                    t1=self.T1,
                    t2=self.T2,
                    t3=self.T3,
                    t4=self.T4,
                    lmbda=self._lambda,
                )

            implementations["piqa"] = piqa_fsim
        except Exception:
            logger.warning("piqa or it's FSIM implementation is not available, skipping piqa implementation of FSIM")

        return implementations

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        arrow = self._arrow_indicating_optimum()

        rgb_info = " using RGB definition" if self.use_rgb else " "
        return (
            f"{self.name} ({self.abbreviation}) {arrow}{rgb_info} with parameters: "
            f"{self.T1=}, {self.T2=}, {self.T3=}, {self.T4=}, {self._lambda=}, "
            f"{self.scales=}, {self.orientations=}, {self.minimal_wavelength=}, "
            f"{self.scaling_factor=}, {self.sigma_f=}, {self.delta_theta=}, "
            f"{self.noise_threshold_factor=}, {self.epsilon=}"
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
            "use_rgb": self.use_rgb,
        }

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to FSIM, such as checking for valid pixel
        value ranges and dimensions.

        Warns if all pixel values in both image and reference are in the range [0, 1],
        which may indicate that the images are not correctly scaled for FSIM, which
        expects pixel values in the range [0, 255].

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :raises ValueError: If the input image contains pixel values outside the range
            [0, 255].
        :raises ValueError: If the reference image contains pixel values outside the
            range [0, 255].
        :raises ValueError: If use_rgb is True but the images do not have 3 channels,
            which is required for the RGB definition of FSIM.

        """
        # Input checks specific to FSIM
        if torch.any(image < 0) or torch.any(image > 255):
            raise ValueError("Input image contains pixel values outside the range [0, 255].")

        if torch.any(reference < 0) or torch.any(reference > 255):
            raise ValueError("Reference image contains pixel values outside the range [0, 255].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            logger.warning(
                "It hase been detected that all pixel values in both image and reference are in the range [0, 1]. "
                "FSIM expects pixel values in the range [0, 255]. Please ensure that your input images are "
                "correctly scaled for accurate computation of FSIM."
            )

        if self.use_rgb:
            channel_dim = self.expected_dimensions.index(Dimension.CHANNEL)
            if image.shape[channel_dim] != 3:
                raise ValueError(
                    f"Images have {image.shape[channel_dim]} channels. FSIM expects 3 (RGB) channels, "
                    "when use_rgb is set to True. Instead you might want to use the grayscale definition"
                    "(use_rgb=False) and apply it channel wise."
                )

    def _subsample(self, image: torch.Tensor, kernel_size: int = 2) -> torch.Tensor:
        """Subsample the input image by a factor of k (default k=2) using a mean filter
        and dyadic subsampling. This simulates the typical distance between an image
        and its viewer in psychophysical experiments as described in the original
        publication.

        If use_rgb is True and the input image has 3 channels, the subsampling is
        applied to each channel separately and then the subsampled channels are stacked
        back together.

        :param image: The input 2D image to be subsampled, shape (H, W), or (C, H, W)
            if use_rgb is True and the image has 3 channels.
        :type image: torch.Tensor
        :return: The subsampled image, shape (H/k, W/k), or (H/k+1, W/k+1) if the input
            dimensions are odd. If use_rgb is True and the input image has 3 channels,
            the output shape will be (C, H/k, W/k) or (C, H/k+1, W/k+1) if the input
            dimensions are odd.
        :rtype: torch.Tensor

        """

        # apply subsampling to each channel separately and stack back together
        if self.use_rgb and image.shape[self.expected_dimensions.index(Dimension.CHANNEL)] == 3:
            subsampled_channels = [self._subsample(image[channel], kernel_size=kernel_size) for channel in range(3)]
            return torch.stack(subsampled_channels, dim=self.expected_dimensions.index(Dimension.CHANNEL))

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

    def _phase_congruency(
        self,
        image: torch.Tensor,
        scales: int = 4,
        orientations: int = 4,
        minimal_wavelength: int = 6,
        scaling_factor: int = 2,
        sigma_f: float = 0.55,
        delta_theta: float = 1.2,
        noise_threshold_factor: float = 2.0,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """Compute the phase congruency map of an image.

        This works by convolving the input image with a set of log Gabor filters at
        multiple scales and orientations, and then computing the phase congruency based
        on the responses of these filters. The phase congruency is a measure of how
        well the local phase information in the image is aligned across different
        scales and orientations, which is thought to correspond to perceptually
        important features in the image.

        First, a set of log Gabor filters is constructed in the frequency domain. Each
        filter is defined as the product of a radial component, which controls the
        frequency band the filter responds to, and an angular component, which controls
        the orientation the filter responds to. The radial component is designed to be
        a bandpass filter that responds to a specific range of frequencies, while the
        angular component is designed to be a directional filter that responds to
        features at a specific orientation.

        Secondly, the input image is convolved with each of the filters to obtain the
        filter responses. The even and odd symmetric responses are used to compute the
        local energy and amplitude at each location in the image. Finally, the phase
        congruency is computed based on the local energy and amplitude, with a noise
        compensation step to suppress responses that are likely dominated by noise.

        Adapted from Kovesi, Peter. "Image features from phase congruency."
        Videre: Journal of computer vision research 1.3 (1999): 1-26.

        :param image: The input 2D image for which to compute the phase congruency,
            shape (H, W)
        :type image: torch.Tensor
        :param scales: Number of wavelet scales, default is 4 as in original
            implementation
        :type scales: int
        :param orientations: Number of filter orientations, default is 4 as in original
            implementation
        :type orientations: int
        :param minimal_wavelength: The wavelength of the smallest scale filter, default
            is 6 as in original implementation
        :type minimal_wavelength: int
        :param scaling_factor: The scaling factor between successive filters, default
            is 2 as in original implementation
        :type scaling_factor: int
        :param sigma_f: The ratio of the standard deviation of the Gaussian describing
            the log Gabor filter's transfer function in the frequency domain to the
            filter center frequency, default is 0.55 as in original implementation
        :type sigma_f: float
        :param delta_theta: The ratio of the angular interval between filter
            orientations to the standard deviation of the Gaussian describing the log
            Gabor filter's transfer function in the frequency domain, default is 1.2 as
            in original implementation
        :type delta_theta: float
        :param noise_threshold_factor: Number of standard deviations above the noise
            mean for the noise compensation threshold, default is 2.0 as in original
            implementation. Below this threshold the response is considered to be
            dominated by noise and is suppressed.
        :type noise_threshold_factor: float
        :param epsilon: A small constant to avoid division by zero, default is 1e-4 as
            in original implementation
        :type epsilon: float
        :return: The phase congruency map of the input image, shape (H, W)
        :rtype: torch.Tensor

        """

        ### Filter construction ###

        # Calculate standard deviation of the angular Gaussian function used to construct filters in frequency domain
        theta_sigma = torch.pi / orientations / delta_theta

        # Pre-compute stuff to speed up filter construction
        grid_x, grid_y = self._get_meshgrid_like(image)
        radius = torch.sqrt(grid_x**2 + grid_y**2)  # normalized radius from the center of the frequency plane,
        theta = torch.atan2(-grid_y, grid_x)  # polar angle

        radius = torch.fft.ifftshift(radius)  # shift the zero frequency component to the center of the spectrum
        theta = torch.fft.ifftshift(theta)
        radius[0, 0] = 1  # to avoid taking log(0) the value at the zero frequency is set to 1

        sin_theta = torch.sin(theta).unsqueeze(0)
        cos_theta = torch.cos(theta).unsqueeze(0)

        ## Radial filter component, which controls the frequency band the filter responds to

        # Low-pass filter that is as large as possible, yet falls away to zero at the boundaries. Log Gabor filters are
        # multiplied by this low-pass filter to ensure that they have no response at the boundaries of the frequency
        # plane, which would otherwise cause artefacts in the spatial domain. The cut-off frequency/radius (.45) and
        # sharpness (15) for the low-pass filter are set according to the original implementation.
        low_pass_filter = self._low_pass_filter_like(image, 0.45, 15)

        center_frequencies = 1.0 / (minimal_wavelength * scaling_factor ** torch.arange(scales, device=image.device))
        log_sigma_f = torch.log(torch.tensor(sigma_f, device=image.device))
        log_gabor = low_pass_filter * torch.exp(
            (-(torch.log(radius / (center_frequencies[:, None, None])) ** 2) / (2 * log_sigma_f**2))
        )
        log_gabor[:, 0, 0] = 0  # undo radius fudge

        ## Angular filter component, which controls the orientation the filter responds to

        # Compute the angular distances for each filter orientation
        angles = torch.arange(orientations, device=image.device)[:, None, None] * torch.pi / orientations
        differences_in_sine = sin_theta * torch.cos(angles) - cos_theta * torch.sin(angles)
        differences_in_cosine = cos_theta * torch.cos(angles) + sin_theta * torch.sin(angles)
        absolute_angular_differences = torch.abs(torch.atan2(differences_in_sine, differences_in_cosine))
        spreads = torch.exp(-(absolute_angular_differences**2) / (2 * theta_sigma**2))

        ### Compute Phase Congruency ###
        image_fourier_transformed = torch.fft.fft2(image)[
            None, None, :, :
        ]  # shape (1, 1, H, W) to broadcast with filters of shape (scales, orientations, H, W)

        filters = (
            log_gabor[:, None, :, :] * spreads[None, :, :, :]
        )  # multiply  radial and angular components to get the filter
        ifft_filters = torch.fft.ifft2(filters, dim=(-2, -1)).real * torch.sqrt(
            torch.prod(torch.tensor(image.shape))
        )  # note rescaling to match power

        # convolve image with filter to get even and odd filter responses in spatial domain
        even_odd = torch.fft.ifft2(image_fourier_transformed * filters, dim=(-2, -1))

        # compute energy from filter responses
        even = even_odd.real
        odd = even_odd.imag

        sum_even = even.sum(dim=0)
        sum_odd = odd.sum(dim=0)

        total_energy = torch.sqrt(sum_even**2 + sum_odd**2) + epsilon
        mean_even = sum_even / total_energy
        mean_odd = sum_odd / total_energy

        energy = torch.sum(even * mean_even + odd * mean_odd - torch.abs(even * mean_odd - odd * mean_even), dim=0)

        ## compensate for noise ##

        # estimate noise power from enery and filter responses at smalles scale
        median_squared_energy = (torch.abs(even_odd[0]) ** 2).reshape(orientations, -1).median(dim=-1).values
        mean_squared_energy = median_squared_energy / torch.log(torch.tensor(2, device=image.device))
        mean_squared_filter = torch.sum(filters[0] ** 2, dim=(-2, -1))
        noise_power = mean_squared_energy / mean_squared_filter

        # estimate total energy squared due to noise
        estimated_total_enery_squared_due_to_noise = 2.0 * noise_power * torch.sum(
            ifft_filters**2, dim=(0, -2, -1)
        ) + 4.0 * noise_power * torch.sum(
            0.5 * (ifft_filters.sum(dim=0) ** 2 - (ifft_filters**2).sum(dim=0)), dim=(-2, -1)
        )

        # determine noise threshold T
        tau = torch.sqrt(estimated_total_enery_squared_due_to_noise / 2.0)
        pi_tensor = torch.tensor(torch.pi, device=image.device)
        estimated_noise_energy = tau * torch.sqrt(pi_tensor / 2.0)
        estimated_noise_energy_sigma = torch.sqrt((2 - pi_tensor / 2.0) * tau**2)

        T = estimated_noise_energy + noise_threshold_factor * estimated_noise_energy_sigma

        # empirically the noise level seems to be overestimated,
        # therefore the threshold is scaled down by a factor of 1.7 as in original implementation
        T = T / 1.7

        energies = torch.sum(torch.clamp(energy - T[:, None, None], min=0.0), dim=0)
        amplitudes = torch.sum(torch.abs(even_odd), dim=(0, 1))
        return energies / amplitudes

    def _get_meshgrid_like(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a meshgrid of coordinates for the given image.

        :param image: The input image for which to create the meshgrid, shape (H, W)
        :type image: torch.Tensor
        :return: A tuple containing the x and y coordinate grids, each of the same
            shape as image.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        height, width = image.shape
        device = image.device
        dtype = image.dtype

        if width % 2:  # Odd
            yrange = torch.arange(-(width - 1) / 2, (width - 1) / 2 + 1, device=device, dtype=dtype) / (width - 1)
        else:  # Even
            yrange = torch.arange(-width / 2, width / 2, device=device, dtype=dtype) / width

        if height % 2:  # Odd
            xrange = torch.arange(-(height - 1) / 2, (height - 1) / 2 + 1, device=device, dtype=dtype) / (height - 1)
        else:  # Even
            xrange = torch.arange(-height / 2, height / 2, device=device, dtype=dtype) / height

        x, y = torch.meshgrid(xrange, yrange, indexing="ij")
        return x, y

    def _low_pass_filter_like(self, image: torch.Tensor, cutoff: float, sharpness: float) -> torch.Tensor:
        """Create a low-pass filter in the frequency domain for the given image.

        :param image: The input image for which to create the filter, shape (H, W)
        :type image: torch.Tensor
        :param cutoff: The cutoff frequency of the low-pass filter, specified as a
            fraction of the Nyquist frequency (0.5 corresponds to the Nyquist
            frequency).
        :type cutoff: float
        :param sharpness: The sharpness of the filter's transition band. Higher values
            result in a sharper transition.
        :type sharpness: float
        :return: The low-pass filter in the frequency domain, shape (H, W)
        :rtype: torch.Tensor

        """

        if cutoff < 0 or cutoff > 0.5:
            raise ValueError(
                "Cutoff frequency must be in the range [0, 0.5], where 0.5 corresponds to the Nyquist frequency."
            )

        if sharpness < 1 or not isinstance(sharpness, int):
            raise ValueError("Sharpness must be a integer >= 1")

        x, y = self._get_meshgrid_like(image)
        radius = torch.sqrt(x**2 + y**2)
        return torch.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * sharpness)))
