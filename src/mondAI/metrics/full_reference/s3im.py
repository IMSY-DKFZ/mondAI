# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from skimage.filters import threshold_local
from skimage.metrics import structural_similarity

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric


class S3IM(FullReferenceMetric):
    """Sparse Structural Similarity Index (S3IM)

    For sparse images, i.e. many pixels are uninformative background, this version of SSIM focussed more on the
    informative content by masking out the uninformative background. The mask is determined by adaptive thresholding,
     which is controlled by a sensitivity parameter.

    Implementation adapted from original numpy implementation by Melle Van Der Brugge, which is available at:
    https://github.com/MellevdB/photoacoustic-image-quality-assessment/blob/main/src/evaluation/metrics/fr.py#L21
    (licence: MIT)

    Original publication:
    Joseph, Francis Kalloor.
    "Sparse structural similarity index (S3IM): a new full reference image quality metric for photoacoustic imaging."
    Photons Plus Ultrasound: Imaging and Sensing 2025. Vol. 13319. SPIE, 2025.

    Additional publication:
    Van Der Brugge, Melle, Kalloor Joseph Francis, and Navchetan Awasthi.
    "Assessing image quality in photoacoustic imaging: A metric-based and deep learning-based evaluation."
    Photoacoustics (2026): 100800.

    """

    @property
    def name(self) -> str:
        return "Sparse Structural Similarity Index"

    @property
    def abbreviation(self) -> str:
        return "S3IM"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def scaling_factor(self) -> float:
        return 1.0

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, sensitivity: float = 0.5) -> None:
        """Initialize the Metric Template.

        The constructor should include checks for any parameters that the metric uses
        to ensure they are within valid ranges or meet certain conditions. For example,
        if the metric has a parameter that must be non-negative, this method should
        check that condition and raise a ValueError if it is not met. This helps to
        prevent invalid configurations of the metric that could lead to incorrect
        results or errors during computation.

        Store all parameters as instance attributes with the same name so that they can
        be included in the fingerprint and string representation of the metric. This
        ensures that the metric's configuration is fully captured and can be reproduced
        accurately.

        :param sensitivity: Sensitivity parameter for adaptive thresholding, which
            influences the threshold level used to create masks for the images. Should
            be non-negative, default is 0.5.
        :type sensitivity: float

        """
        super().__init__()
        self.sensitivity = sensitivity

        if self.sensitivity < 0:
            raise ValueError(f"sensitivity must be non-negative, but got {self.sensitivity}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the metric between image and reference.

        Please only perform metric specific checks in this method. General checks for
        input images are already performed in the base class.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :return: The computed metric score.
        :rtype: torch.Tensor

        """

        neighborhood_size = (image.shape[0] // 16) * 2 + 1

        # determine thresholds adaptively
        level_image = threshold_local(
            image.numpy(force=True), neighborhood_size, method="gaussian", offset=self.sensitivity
        )
        level_reference = threshold_local(
            reference.numpy(force=True), neighborhood_size, method="gaussian", offset=self.sensitivity
        )
        # FIXME: Convert threshold_local into torch code

        mask_image = image > torch.tensor(level_image, device=image.device)
        mask_reference = reference > torch.tensor(level_reference, device=reference.device)
        mask = mask_image & mask_reference

        masked_image = image * mask
        masked_reference = reference * mask

        # FIXME: Convert skimage's structural_similarity into torch code
        ssim_map = torch.tensor(
            structural_similarity(
                masked_image.numpy(force=True),
                masked_reference.numpy(force=True),
                data_range=image.max().item() - image.min().item(),
                full=True,
            )[1],
            device=image.device,
        )
        s3im_map = ssim_map * mask

        return torch.sum(s3im_map) / torch.sum(mask) if torch.any(mask) else torch.tensor(0.0)

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        """Override this method in subclasses to register other implementations of the
        metric for comparison. Use the `_register_implementation` helper method to add
        implementations to the internal dictionary. These implementations will be used
        when compare_implementations is True.

        Store reference implementations in the `third_party` submodule of the metrics module,
        and import them here to register them for comparison.

        """

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} with sensitivity={self.sensitivity}"
        )
