from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch

from mondAI.metrics.base import Metric
from mondAI.settings import settings
from mondAI.utils.checks import (
    check_same_shape,
    check_same_type,
)


class FullReferenceMetric(Metric, ABC):
    """Abstract base class for metrics that require a reference image."""

    def __call__(
        self,
        image: np.ndarray | torch.Tensor,
        reference: np.ndarray | torch.Tensor,
        dims: Sequence[str] = settings.default_dims,
    ) -> float | torch.Tensor | np.ndarray:
        """Compute the metric between image and corresponding reference."""

        # Full reference specific input checks
        check_same_type(image, reference)
        check_same_shape(image, reference)

        return self._call_with_vectorization(image, reference, dims=dims)

    def _compute_vmapped(self, *reshaped_images: torch.Tensor) -> torch.Tensor:
        """Function to compute the metric in a vectorized manner over flattened items.
        Returns a 1D tensor (N,).

        :param reshaped_images: Tuple of tensors containing the images and references, each of shape (N, ...).
        :type reshaped_images: tuple[torch.Tensor, torch.Tensor]
        :return: Tensor containing metric scores for each item in the batch.
        :rtype: torch.Tensor

        """
        images, references = reshaped_images
        return torch.vmap(self._compute)(images, references)

    @abstractmethod
    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Internal method to compute the metric.

        To be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses should implement this method.")
