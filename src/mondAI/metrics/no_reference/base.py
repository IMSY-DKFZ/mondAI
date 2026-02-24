from abc import ABC, abstractmethod
from typing import Sequence

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
    ) -> float | torch.Tensor | np.ndarray:
        """Compute the metric for the given image."""

        return self._call_with_vectorization(image, dims=dims)

    def _compute_vmapped(self, *reshaped_images: torch.Tensor) -> torch.Tensor:
        """Function to compute the metric in a vectorized manner over flattened items.
        Returns a 1D tensor (N,).

        :param reshaped_images: Tuple of tensors containing the images and references, each of shape (N, ...).
        :type reshaped_images: tuple[torch.Tensor]
        :return: Tensor containing metric scores for each item in the batch.
        :rtype: torch.Tensor

        """
        (images,) = reshaped_images
        return torch.vmap(self._compute)(images)

    @abstractmethod
    def _compute(self, image: torch.Tensor) -> float | torch.Tensor:
        """Internal method to compute the metric.

        To be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses should implement this method.")
