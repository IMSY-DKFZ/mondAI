from typing import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric

logger = get_logger()


class PSNR(FullReferenceMetric):
    r"""Peak Signal-to-Noise Ratio (PSNR).

    Peak Signal-to-Noise Ratio (PSNR) is a full-reference fidelity measure
    derived from the mean squared error (MSE). It expresses the ratio between the
    maximum possible signal value and the reconstruction error on a logarithmic decibel
    scale, ranging from 0 to infinity. Higher PSNR values indicate smaller reconstruction errors,
    and identical images (MSE is zero) yield an infinite PSNR value.

    The metric is defined as

    .. math::
        \operatorname{PSNR}(x, y) = 10 \log_{10}\left(\frac{L^2}{\operatorname{MSE}(x, y)}\right),

    where

    .. math::
        \operatorname{MSE}(x, y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2,

    :math:`L` is the dynamic range of the image intensities, and :math:`N` is the
    number of pixels. Note that :math:`L` is not necessarily the maximum pixel intensity value in the
    given images, but the maximum possible intensity value in the given dynamic range.
    This implementation expects images in ``[0, dynamic_range]`` and defaults to ``dynamic_range=255``.

    Reference implementation used for comparison and API alignment:
    (https://github.com/scikit-image/scikit-image/blob/main/src/skimage/metrics/simple_metrics.py
    commit: d0b36ad1651ee2d7f2a46a84b06ba593a0885c6e, License: BSD-3-Clause).

    """

    @property
    def name(self) -> str:
        return "Peak Signal-to-Noise Ratio"

    @property
    def abbreviation(self) -> str:
        return "PSNR"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, dynamic_range: float = 255.0) -> None:
        """Initialize PSNR.

        :param dynamic_range: Dynamic range ``L`` of the images, which is given by the difference between the maximum
        and minimum possible values (255.0 for unit8, 1.0 for normalized images), default is 255.0.
        :type dynamic_range: float
        :raises ValueError: If ``dynamic_range`` is not positive.

        """
        super().__init__()
        self.dynamic_range = dynamic_range

        if self.dynamic_range <= 0:
            raise ValueError("dynamic_range must be positive.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute PSNR between image and reference."""
        self._input_checks(image, reference)

        mse = torch.mean((image - reference) ** 2)
        if torch.isclose(mse, torch.tensor(0.0, device=mse.device, dtype=mse.dtype)):
            return torch.tensor(torch.inf, device=image.device, dtype=image.dtype)

        return 10.0 * torch.log10((self.dynamic_range**2) / mse)

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the PSNR metric."""
        implementations = {}

        try:
            from skimage.metrics import peak_signal_noise_ratio as psnr_skimage

            def skimage_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # scikit-image operates on NumPy arrays and is the closest API match.
                value = psnr_skimage(
                    reference.cpu().numpy(),
                    image.cpu().numpy(),
                    data_range=self.dynamic_range,
                )
                return torch.tensor(value, device=image.device, dtype=image.dtype)

            implementations["scikit-image"] = skimage_psnr
        except Exception:
            logger.warning("scikit-image is not available, skipping scikit-image implementation of PSNR.")

        try:
            from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_torchmetrics

            def torchmetrics_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # torchmetrics expects NCHW tensors for image metrics.
                return psnr_torchmetrics(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    data_range=self.dynamic_range,
                )

            implementations["torchmetrics"] = torchmetrics_psnr
        except Exception:
            logger.warning("torchmetrics is not available, skipping torchmetrics implementation of PSNR.")

        try:
            import tensorflow as tf

            def tensorflow_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # TensorFlow expects NHWC tensors.
                value = tf.image.psnr(
                    image.cpu().numpy()[None, ..., None],
                    reference.cpu().numpy()[None, ..., None],
                    max_val=self.dynamic_range,
                ).numpy()
                return torch.tensor(value.item(), device=image.device, dtype=image.dtype)

            implementations["tensorflow"] = tensorflow_psnr
        except Exception:
            logger.warning("tensorflow is not available, skipping tensorflow implementation of PSNR.")

        try:
            from piq import psnr

            def piq_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # PIQ expects NCHW tensors.
                return psnr(
                    image.unsqueeze(0).unsqueeze(0),
                    reference.unsqueeze(0).unsqueeze(0),
                    data_range=self.dynamic_range,
                    reduction="mean",
                    convert_to_greyscale=False,
                )

            implementations["piq"] = piq_psnr
        except Exception:
            logger.warning("piq is not available, skipping piq implementation of PSNR.")

        try:
            from piqa import PSNR as PIQAPSNR

            piqa_metric = PIQAPSNR(value_range=self.dynamic_range, reduction="mean")

            def piqa_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # PIQA exposes PSNR as a module and expects NCHW tensors.
                return piqa_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0))

            implementations["piqa"] = piqa_psnr
        except Exception:
            logger.warning("piqa is not available, skipping piqa implementation of PSNR.")

        try:
            from sewar.full_ref import psnr as sewar_psnr

            def sewar_psnr_impl(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # sewar operates on NumPy arrays.
                value = sewar_psnr(reference.cpu().numpy(), image.cpu().numpy(), MAX=self.dynamic_range)
                return torch.tensor(value, device=image.device, dtype=image.dtype)

            implementations["sewar"] = sewar_psnr_impl
        except Exception:
            logger.warning("sewar is not available, skipping sewar implementation of PSNR.")

        try:
            from monai.metrics import PSNRMetric

            monai_metric = PSNRMetric(max_val=self.dynamic_range)

            def monai_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # MONAI expects tensors in NCHW format.
                return torch.as_tensor(
                    monai_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0)),
                    device=image.device,
                )

            implementations["monai"] = monai_psnr
        except Exception:
            logger.warning("monai is not available, skipping monai implementation of PSNR.")

        try:
            from deepinv.loss.metric import PSNR as DeepInvPSNR

            deepinv_metric = DeepInvPSNR(max_pixel=self.dynamic_range)

            def deepinv_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # deepinv expects NCHW tensors.
                return torch.as_tensor(
                    deepinv_metric(
                        image.unsqueeze(0).unsqueeze(0),
                        reference.unsqueeze(0).unsqueeze(0),
                        max_pixel=self.dynamic_range,
                    ),
                    device=image.device,
                )

            implementations["deepinv"] = deepinv_psnr
        except Exception:
            logger.warning("deepinv is not available, skipping deepinv implementation of PSNR.")

        try:
            from mondAI.metrics.third_party.medimetrics.psnr import PSNR as MediMetricsPSNR

            medimetric = MediMetricsPSNR()

            def medimetrics_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # medimetrics exposes a NumPy-based compute method.
                value = medimetric.compute(reference.cpu().numpy(), image.cpu().numpy())
                return torch.tensor(value, device=image.device, dtype=image.dtype)

            implementations["medimetrics"] = medimetrics_psnr
        except Exception:
            logger.warning("medimetrics is not available, skipping medimetrics implementation of PSNR.")

        return implementations

    def __str__(self) -> str:
        """Full text representation of the PSNR metric."""
        arrow = self._arrow_indicating_optimum()
        return f"{self.name} ({self.abbreviation}) {arrow} with {self.dynamic_range=}"

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to PSNR."""
        if torch.any(image < 0) or torch.any(image > self.dynamic_range):
            raise ValueError(f"Input image contains pixel values outside the range [0, {self.dynamic_range}].")

        if torch.any(reference < 0) or torch.any(reference > self.dynamic_range):
            raise ValueError(f"Reference image contains pixel values outside the range [0, {self.dynamic_range}].")

        if torch.all(image >= 0) and torch.all(image <= 1) and torch.all(reference >= 0) and torch.all(reference <= 1):
            if self.dynamic_range == 255.0:
                logger.warning(
                    "It has been detected that all pixel values in both image and reference are in the range [0, 1]. "
                    "PSNR defaults to dynamic_range=255. Please ensure that your input images are correctly scaled or "
                    "set dynamic_range=1.0 for normalized inputs."
                )
