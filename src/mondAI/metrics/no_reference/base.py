from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import torch

from mondAI.metrics.base import Metric
from mondAI.settings import settings


class NoReferenceMetric(Metric, ABC):
    """Abstract base class for metrics that do not require a reference image."""

    def __call__(
        self,
        image: np.ndarray | torch.Tensor,
        dims: Sequence[str] = settings.default_dims,
        compare_implementations: bool = False,
    ) -> float | torch.Tensor | np.ndarray | dict[str, float | torch.Tensor | np.ndarray]:
        """Compute the metric for the given image.

        If compare_implementations is True, also compute the metric using other
        implementations.

        :param image: The image to be evaluated.
        :type image: np.ndarray | torch.Tensor
        :param dims: The dimensions of the input image, e.g. ("H", "W") for 2D images,
            ("C", "H", "W") for RGB images
        :type dims: Sequence[str]
        :param compare_implementations: Whether to compute the metric using other
            implementations for comparison.
        :type compare_implementations: bool
        :return: The computed metric score, or a dictionary containing the scores from
            all implementations if compare_implementations is True.
        :rtype: float | torch.Tensor | np.ndarray | dict[str, float | torch.Tensor |
            np.ndarray]

        """

        if compare_implementations:
            scores = {
                library_name: self._call_pipeline(image, dims=dims, compute_function=implementation)
                for library_name, implementation in self._other_implementations().items()
            }
            scores["mondAI"] = self._call_pipeline(image, dims=dims, compute_function=self._compute)
            return scores

        else:
            return self._call_pipeline(image, dims=dims, compute_function=self._compute)

    def _compute_iteratively(
        self, *reshaped_images: torch.Tensor, compute_function: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Function to compute the metric in an iterative manner over flattened items.
        Returns a 1D tensor (N,).

        :param reshaped_images: Tuple of tensors containing the images and references, each of shape (N, ...).
        :type reshaped_images: tuple[torch.Tensor]
        :param compute_function: The function to compute the metric for a single image.
        :type compute_function: Callable[torch.Tensor, torch.Tensor]
        :return: Tensor containing metric scores for each item in the batch.
        :rtype: torch.Tensor

        """
        (images,) = reshaped_images
        return torch.stack([compute_function(image) for image in images.unbind(0)])

    @abstractmethod
    def _compute(self, image: torch.Tensor) -> float | torch.Tensor:
        """Internal method to compute the metric.

        To be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses should implement this method.")
